import os
import yaml

import cv2
import matplotlib.pyplot as plt
import numpy as np

from nerfstudio.utils.rich_utils import CONSOLE, status


def get_calibration_target_circle_centers():
    # The fixed marker coordinates here are specified in OpenCV coords, though: x right, y down, z=0
    # diameter = 1.5  # cm
    c_c = 1.5 + 2.3  # centre-centre (vertical and horizontal)
    # c_c = 1.
    dist = 0.  # planar calibration points
    objpoints = []
    # for col in range(10, -1, -1):
    for col in range(11):
        ypt = c_c / 2 * col
        for row in range(4):
            # for row in range(3, -1, -1):
            if col % 2 == 0:
                xpt = c_c * row
            else:
                xpt = c_c * row + c_c / 2
            # objpoints.append([xpt, ypt, dist])
            objpoints.append([ypt, xpt, dist])
    return np.array(objpoints, dtype=np.float32)


def circle_detect(captured_img, num_circles=(4, 11), show_preview=False):
    """Detects the circle of a circle board pattern

    :param captured_img: captured image
    :param num_circles: a tuple of integers, [circles per column x circles per row]
    :param show_preview: boolean, default True
    :return: corners: These corners will be placed in an order (from left-to-right, top-to-bottom)
             found_dots: boolean, indicating success of calibration
    """

    # Binarization
    # org_copy = org.copy() # Otherwise, we write on the original image!
    # img = (captured_img.copy() * 255).astype(np.uint8)
    img = captured_img.copy()
    # img = 255 - captured_img.copy()
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = cv2.medianBlur(img, 15)
    img = cv2.medianBlur(img, 5)
    img_gray = img.copy()

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 121, 10)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # img = 255 - img
    # img_gray = img_gray

    # Blob detection
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.filterByColor = True
    params.minThreshold = 128

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 300

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.8

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.80

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)

    # Detecting keypoints
    # this is redundant for what comes next, but gives us access to the detected dots for debug
    keypoints = detector.detect(img)

    # found_dots, centers = cv2.findCirclesGrid(img, patternSize=num_circles,
    #                                           blobDetector=detector, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

    found_dots, centers = cv2.findCirclesGrid(img,
                                              patternSize=num_circles,
                                              blobDetector=detector,
                                              flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
    # print(f"detected {np.shape(keypoints)} keypoints / found_dots: {found_dots}")

    # NOTE: For now, assert that we find calibration pattern for all our calibration images.
    #  In the future, might want to lift this restriction and just match images for which we
    #  find calibration pattern.
    assert found_dots
    assert len(keypoints) == num_circles[0] * num_circles[1]

    if show_preview:
        # Drawing the keypoints
        captured_img = cv2.drawChessboardCorners(captured_img, num_circles, centers, found_dots)
        img_gray = cv2.drawKeypoints(img_gray,
                                     keypoints,
                                     np.array([]),
                                     (0, 255, 0),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # print(f"Num keypoints: {np.shape(keypoints)}")
        # print(f"Num centers: {np.shape(centers)}")

        fig = plt.figure()

        ax = fig.add_subplot(223)
        ax.imshow(img_gray, cmap='gray')
        ax.set_title("img_gray")

        ax2 = fig.add_subplot(221)
        ax2.imshow(img, cmap='gray')
        ax2.set_title("img")

        ax3 = fig.add_subplot(222)
        ax3.imshow(captured_img, cmap='gray')
        ax3.set_title("captured_img")

        ax4 = fig.add_subplot(224)
        ax4.imshow(captured_img, cmap='gray')
        ax4.set_title("captured_img")

        if centers is not None:
            for kk in range(len(centers)):
                plt.plot(centers[kk, 0, 0], centers[kk, 0, 1], 'gx')
                ax4.annotate(f"{kk}", (centers[kk, 0, 0] + 5, centers[kk, 0, 1]), fontsize=8, color="r")

        # plt.savefig('output.png')
        plt.show()

    return centers, found_dots


def estimate_intrinsics(image_file_names=[],
                        marker_coordinates=[],
                        imgsize=(320, 320),
                        intrinsic_calibration_mode=2,
                        force_tangential_distortion_coeffs_to_zero=True,
                        force_radial_distortion_coeff_K1_K2_to_zero=False,
                        force_radial_distortion_coeff_K3_to_zero=True
                        ):
    """ Given a set of image filenames, detect marker coordinates and estimate intrinsic camera parameters.

    input parameters:
      intrinsic_calibration_mode
          mode 0: don't fix focal length or principle point
          mode 1: fix principle point to image center but don't fix focal length
          mode 2: don't fix principle point but fix focal length of x and y to be the same
          mode 3: fix principle point to image center and fix focal length of x and y to be the same
          mode 4: fix focal length of x and y to be the same but don't use initial guess
    """
    #####################################################################
    # 1. find markers in images

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    valid_image_idx = np.zeros(len(image_file_names))

    for k in range(len(image_file_names)):

        # load image
        img = cv2.imread(image_file_names[k])

        # detect 2D circle centers
        (corners, found_dots) = circle_detect(img.copy(), show_preview=False)

        if found_dots == True:
            # add pairs of reference points and detected 2D circle centers to lists
            objpoints.append(marker_coordinates)
            imgpoints.append(corners)
            valid_image_idx[k] = 1

    #####################################################################
    # 2. run OpenCV's calibration

    # these flags are the options for the calibration
    calibration_flags = 0

    # force tangential distortion to be 0
    if force_tangential_distortion_coeffs_to_zero:
        calibration_flags += cv2.CALIB_ZERO_TANGENT_DIST

    # force radial coefficient K3 to be 0
    if force_radial_distortion_coeff_K3_to_zero:
        calibration_flags += cv2.CALIB_FIX_K3

    # force radial coefficients K1 and K2 to be 0
    if force_radial_distortion_coeff_K1_K2_to_zero:
        calibration_flags += cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2

    # no initial guess
    if intrinsic_calibration_mode == 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgsize, None, None,
                                                           flags=calibration_flags)
    elif intrinsic_calibration_mode == 4:
        calibration_flags += cv2.CALIB_FIX_ASPECT_RATIO
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgsize, None, None,
                                                           flags=calibration_flags)

    # initial guess of camera matrix (the specific values of the focal length are not important, but their ratio will be kept)
    else:
        mtx = np.array([[1000.0, 0., imgsize[0] / 2.0],
                        [0., 1000.0, imgsize[1] / 2.0],
                        [0., 0., 1.]], dtype=np.float32)

        if intrinsic_calibration_mode == 1:
            calibration_flags += (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT)
        elif intrinsic_calibration_mode == 2:
            calibration_flags += (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO)
        elif intrinsic_calibration_mode == 3:
            calibration_flags += (
                    cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT)

            # with initial guess
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgsize, mtx, None,
                                                           flags=calibration_flags)

    #####################################################################
    # 3. compute reprojection error

    mean_error = 0.0
    num_valid_images = np.shape(objpoints)[0]
    for k in range(num_valid_images):
        # project 3D marker coordinates
        projected_points, _ = cv2.projectPoints(objpoints[k], rvecs[k], tvecs[k], mtx, dist)

        # evaluate error
        error = cv2.norm(imgpoints[k], projected_points, cv2.NORM_L2) / len(projected_points)
        mean_error += error

    if num_valid_images > 0:
        mean_error /= num_valid_images

    #####################################################################
    # 4. prepare results

    result = {}
    result["camera_matrix"] = mtx
    result["distortion_coeffs"] = dist
    result["rvecs"] = rvecs
    result["tvecs"] = tvecs
    result["rmse"] = mean_error

    return result


def evaluate_intrinsics(image_file_names=[],
                        marker_coordinates=[],
                        imgsize=(320, 320),
                        camera_matrix=np.array((3, 3), dtype=float),
                        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0)
                        ):
    """
    Given a set of image filenames, detect marker coordinates and evaluate error for a
    set of validation images.
    """
    mean_error = 0.0
    num_valid_images = 0
    for k in range(len(image_file_names)):

        # load image
        img = cv2.imread(image_file_names[k])

        # detect 2D circle centers
        (corners, found_dots) = circle_detect(img.copy(), show_preview=False)

        if found_dots == True:
            num_valid_images += 1

            # do an extrinsic calibration of just this image
            ret, rvec, tvec = cv2.solvePnP(marker_coordinates,
                                           corners,
                                           camera_matrix,
                                           distortion_coefficients,
                                           flags=cv2.SOLVEPNP_ITERATIVE)

            # project 3D marker coordinates into 2D image coordinates
            projected_points, _ = cv2.projectPoints(marker_coordinates, rvec, tvec, camera_matrix,
                                                    distortion_coefficients)

            # evaluate error
            error = cv2.norm(corners, projected_points, cv2.NORM_L2) / len(projected_points)
            mean_error += error

    if num_valid_images > 0:
        mean_error /= num_valid_images

    return error


def calibrate_camera(
        folder,
        intrinsic_calibration_mode=2,
        force_tangential_distortion_coeffs_to_zero=False,
        force_radial_distortion_coeff_K1_K2_to_zero=False,
        force_radial_distortion_coeff_K3_to_zero=False,
        save_results=False,
        validate=False,
):
    # get all filenames in this folder
    # files_in_folder_ = os.listdir(folder)
    files_in_folder = []
    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
                files_in_folder.append(os.path.join(folder, f))
    files_in_folder = sorted(files_in_folder)
    # print(files_in_folder)
    # print(f"Found {len(files_in_folder)} files in target folder")

    # image resolution (height, width)
    imgsize = cv2.imread(files_in_folder[0]).shape[:2]

    # 3D coordinates of the circle centers
    #   Note: if the 3 circles of the calibration markers point down in the images, then switch the flag to False
    marker_coordinates = get_calibration_target_circle_centers()

    # run intrinsic calibration
    result = estimate_intrinsics(
        image_file_names=files_in_folder,
        marker_coordinates=marker_coordinates,
        imgsize=imgsize,
        intrinsic_calibration_mode=intrinsic_calibration_mode,
        force_tangential_distortion_coeffs_to_zero=force_tangential_distortion_coeffs_to_zero,
        force_radial_distortion_coeff_K1_K2_to_zero=force_radial_distortion_coeff_K1_K2_to_zero,
        force_radial_distortion_coeff_K3_to_zero=force_radial_distortion_coeff_K3_to_zero
    )

    print("---------------------------")
    print("Camera matrix:")
    print(result["camera_matrix"])
    print(" ")

    print("Distortion Coefficients:")
    print(result["distortion_coeffs"])
    print(" ")

    print(f"Train RMSE: {result['rmse']}")

    if validate:
        validation_error = evaluate_intrinsics(
            image_file_names=files_in_folder,
            marker_coordinates=marker_coordinates,
            camera_matrix=result["camera_matrix"],
            distortion_coefficients=result["distortion_coeffs"],
            imgsize=imgsize
        )

        print(" ")
        print("Validation RMSE:")
        print(validation_error)

    if save_results:  # save to file
        mtx_undistorted, roi_undistorted = cv2.getOptimalNewCameraMatrix(
            result["camera_matrix"], result["distortion_coeffs"], imgsize, 1, imgsize
        )
        focal_length = (mtx_undistorted[0,0] + mtx_undistorted[1,1]) / 2.0

        data = {"camera_matrix": np.asarray(result["camera_matrix"]).tolist(),
                "dist_coeff": np.asarray(result["distortion_coeffs"]).tolist(),
                "focal_length": float(focal_length)}
        with open("calibration.yaml", "w") as f:
            yaml.dump(data, f)

    return result


def calibrate_rgb_thermal(
        rgb_folder,
        thermal_folder,
        intrinsic_calibration_mode=4,
        force_tangential_distortion_coeffs_to_zero=False,
        force_radial_distortion_coeff_K1_K2_to_zero=False,
        force_radial_distortion_coeff_K3_to_zero=False,
):
    """Calibration and relative transform of RGB and thermal cameras.

    Given folders of RGB and thermal images, computes intrinsic matrix and distortion coefficients for RGB and thermal
    cameras and the average relative translation and rotation from RGB to thermal camera.
    """
    result_rgb = calibrate_camera(
        rgb_folder,
        intrinsic_calibration_mode=intrinsic_calibration_mode,
        force_tangential_distortion_coeffs_to_zero=force_tangential_distortion_coeffs_to_zero,
        force_radial_distortion_coeff_K1_K2_to_zero=force_radial_distortion_coeff_K1_K2_to_zero,
        force_radial_distortion_coeff_K3_to_zero=force_radial_distortion_coeff_K3_to_zero,
    )
    result_thermal = calibrate_camera(
        thermal_folder,
        intrinsic_calibration_mode=intrinsic_calibration_mode,
        force_tangential_distortion_coeffs_to_zero=force_tangential_distortion_coeffs_to_zero,
        force_radial_distortion_coeff_K1_K2_to_zero=force_radial_distortion_coeff_K1_K2_to_zero,
        force_radial_distortion_coeff_K3_to_zero=force_radial_distortion_coeff_K3_to_zero,
    )

    tvecs_rgb = np.array(result_rgb["tvecs"])
    tvecs_thermal = np.array(result_thermal["tvecs"])
    rvecs_rgb = result_rgb["rvecs"]
    rvecs_thermal = result_thermal["rvecs"]

    # Get relative transforms between rgb and thermal camera per image
    R_rgb_thermals = []
    R_thermal_rgbs = []
    for i in range(len(rvecs_rgb)):
        R_rgb, R_thermal = (np.identity(3) for _ in range(2))
        R_rgb[:3,:3], _ = cv2.Rodrigues(rvecs_rgb[i])
        R_thermal[:3,:3], _ = cv2.Rodrigues(rvecs_thermal[i])

        R_rgb_thermals.append(R_thermal @ np.linalg.inv(R_rgb))
        R_thermal_rgbs.append(R_rgb @ np.linalg.inv(R_thermal))
    R_rgb_thermals = np.array(R_rgb_thermals)
    R_thermal_rgbs = np.array(R_thermal_rgbs)

    # Compute "average" relative rotation
    mean_R_rgb_thermal = R_rgb_thermals.mean(axis=0)
    U, S, Vh = np.linalg.svd(mean_R_rgb_thermal)
    R_rgb_thermal = U @ Vh

    mean_R_thermal_rgb = R_thermal_rgbs.mean(axis=0)
    U, S, Vh = np.linalg.svd(mean_R_thermal_rgb)
    R_thermal_rgb = U @ Vh

    # Compute "average" relative translation
    t_rgb_thermals = []
    t_thermal_rgbs= []
    for i in range(len(rvecs_rgb)):
        t_rgb_thermals.append(tvecs_thermal[i] - R_rgb_thermals[i] @ tvecs_rgb[i])
        t_thermal_rgbs.append(tvecs_rgb[i] - R_thermal_rgbs[i] @ tvecs_thermal[i])
    t_rgb_thermals = np.array(t_rgb_thermals)
    t_thermal_rgbs = np.array(t_thermal_rgbs)
    t_rgb_thermal = t_rgb_thermals.mean(axis=0)
    t_thermal_rgb = t_thermal_rgbs.mean(axis=0)

    # Build 4x4 transform matrices
    M_rgb_thermal = np.identity(4)
    M_rgb_thermal[:3,3] = t_rgb_thermal.squeeze()
    M_rgb_thermal[:3,:3] = R_rgb_thermal

    M_thermal_rgb = np.identity(4)
    M_thermal_rgb[:3,3] = t_thermal_rgb.squeeze()
    M_thermal_rgb[:3,:3] = R_thermal_rgb

    print("---------------------------")
    np.set_printoptions(suppress=True)
    print("t_thermal_rgbs:")
    print(t_thermal_rgbs.squeeze())
    print("")

    print("---------------------------")
    print(f"Relative rotation:")
    print(R_rgb_thermal)
    print("")

    print("Relative translation:")
    print(t_rgb_thermal.squeeze())
    print("")

    # print("Relative transformation:")
    # print(M_rgb_thermal)

    print("Transforms (1, 0, 0) to:")
    print(M_rgb_thermal @ np.array([1, 0, 0, 1]))

    print("---------------------------")
    print(f"Relative rotation:")
    print(R_thermal_rgb)
    print("")

    print("Relative translation:")
    print(t_thermal_rgb.squeeze())
    print("")

    # print("Relative transformation:")
    # print(M_thermal_rgb)

    print("Transforms (1, 0, 0) to:")
    print(M_thermal_rgb @ np.array([1, 0, 0, 1]))
    print("---------------------------")

    result = {
        "camera_matrix_rgb": result_rgb["camera_matrix"],
        "camera_matrix_thermal": result_thermal["camera_matrix"],
        "distortion_coeffs_rgb": result_rgb["distortion_coeffs"],
        "distortion_coeffs_thermal": result_thermal["distortion_coeffs"],
        "rgb_thermal_transform": M_rgb_thermal,
        "thermal_rgb_transform": M_thermal_rgb,
    }
    return result

