#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import io
import json
import os
import os.path
import re
import csv
import subprocess
from PIL import Image
from math import sqrt, exp, log
from matplotlib import cm
from matplotlib import pyplot as plt

import numpy as np
import skimage

import cv2
import copy

from nerfstudio.utils.rich_utils import CONSOLE, status


class FlirImageExtractor:

    def __init__(self, exiftool_path="exiftool", is_debug=False):
        self.exiftool_path = exiftool_path
        self.is_debug = is_debug
        self.flir_img_filename = ""
        self.image_suffix = "_rgb_image.jpg"
        self.thumbnail_suffix = "_rgb_thumb.jpg"
        self.thermal_suffix = "_thermal.png"
        self.default_distance = 1.0

        # valid for PNG thermal images
        self.use_thumbnail = False
        self.fix_endian = True

        self.rgb_image_np = None
        self.thermal_image_np = None

    pass

    def process_image(self, flir_img_filename):
        """
        Given a valid image path, process the file: extract real thermal values
        and a thumbnail for comparison (generally thumbnail is on the visible spectre)
        :param flir_img_filename:
        :return:
        """
        if self.is_debug:
            print("INFO Flir image filepath:{}".format(flir_img_filename))

        if not os.path.isfile(flir_img_filename):
            raise ValueError("Input file does not exist or this user don't have permission on this file")

        self.flir_img_filename = flir_img_filename

        if self.get_image_type().upper().strip() == "TIFF":
            # valid for tiff images from Zenmuse XTR
            self.use_thumbnail = True
            self.fix_endian = False

        self.rgb_image_np = self.extract_embedded_image()
        self.thermal_image_np = self.extract_thermal_image()

    def get_image_type(self):
        """
        Get the embedded thermal image type, generally can be TIFF or PNG
        :return:
        """
        meta_json = subprocess.check_output(
            [self.exiftool_path, '-RawThermalImageType', '-j', self.flir_img_filename])
        meta = json.loads(meta_json.decode())[0]

        return meta['RawThermalImageType']

    def get_rgb_np(self):
        """
        Return the last extracted rgb image
        :return:
        """
        return self.rgb_image_np

    def get_thermal_np(self):
        """
        Return the last extracted thermal image
        :return:
        """
        return self.thermal_image_np

    def extract_embedded_image(self):
        """
        extracts the visual image as 2D numpy array of RGB values
        """
        image_tag = "-EmbeddedImage"
        if self.use_thumbnail:
            image_tag = "-ThumbnailImage"

        visual_img_bytes = subprocess.check_output([self.exiftool_path, image_tag, "-b", self.flir_img_filename])
        visual_img_stream = io.BytesIO(visual_img_bytes)

        visual_img = Image.open(visual_img_stream)
        visual_np = np.array(visual_img)

        return visual_np

    def extract_thermal_image(self):
        """
        extracts the thermal image as 2D numpy array with temperatures in oC
        """

        # read image metadata needed for conversion of the raw sensor values
        # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
        meta_json = subprocess.check_output(
            [self.exiftool_path, self.flir_img_filename, '-Emissivity', '-SubjectDistance', '-AtmosphericTemperature',
             '-ReflectedApparentTemperature', '-IRWindowTemperature', '-IRWindowTransmission', '-RelativeHumidity',
             '-PlanckR1', '-PlanckB', '-PlanckF', '-PlanckO', '-PlanckR2', '-j'])
        meta = json.loads(meta_json.decode())[0]

        # exifread can't extract the embedded thermal image, use exiftool instead
        thermal_img_bytes = subprocess.check_output([self.exiftool_path, "-RawThermalImage", "-b", self.flir_img_filename])
        thermal_img_stream = io.BytesIO(thermal_img_bytes)

        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        # raw values -> temperature
        subject_distance = self.default_distance
        if 'SubjectDistance' in meta:
            subject_distance = FlirImageExtractor.extract_float(meta['SubjectDistance'])

        if self.fix_endian:
            # fix endianness, the bytes in the embedded png are in the wrong order
            thermal_np = np.vectorize(lambda x: (x >> 8) + ((x & 0x00ff) << 8))(thermal_np)

        raw2tempfunc = np.vectorize(lambda x: FlirImageExtractor.raw2temp(x, E=meta['Emissivity'], OD=subject_distance,
                                                                          RTemp=FlirImageExtractor.extract_float(
                                                                              meta['ReflectedApparentTemperature']),
                                                                          ATemp=FlirImageExtractor.extract_float(
                                                                              meta['AtmosphericTemperature']),
                                                                          IRWTemp=FlirImageExtractor.extract_float(
                                                                              meta['IRWindowTemperature']),
                                                                          IRT=meta['IRWindowTransmission'],
                                                                          RH=FlirImageExtractor.extract_float(
                                                                              meta['RelativeHumidity']),
                                                                          PR1=meta['PlanckR1'], PB=meta['PlanckB'],
                                                                          PF=meta['PlanckF'],
                                                                          PO=meta['PlanckO'], PR2=meta['PlanckR2']))
        thermal_np = raw2tempfunc(thermal_np)
        return thermal_np

    @staticmethod
    def raw2temp(raw, E=1, OD=1, RTemp=20, ATemp=20, IRWTemp=20, IRT=1, RH=50, PR1=21106.77, PB=1501, PF=1, PO=-7340,
                 PR2=0.012545258):
        """
        convert raw values from the flir sensor to temperatures in C
        # this calculation has been ported to python from
        # https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
        # a detailed explanation of what is going on here can be found there
        """

        # constants
        ATA1 = 0.006569
        ATA2 = 0.01262
        ATB1 = -0.002276
        ATB2 = -0.00667
        ATX = 1.9

        # transmission through window (calibrated)
        emiss_wind = 1 - IRT
        refl_wind = 0

        # transmission through the air
        h2o = (RH / 100) * exp(1.5587 + 0.06939 * (ATemp) - 0.00027816 * (ATemp) ** 2 + 0.00000068455 * (ATemp) ** 3)
        tau1 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))
        tau2 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))

        # radiance from the environment
        raw_refl1 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl1_attn = (1 - E) / E * raw_refl1
        raw_atm1 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1
        raw_wind = PR1 / (PR2 * (exp(PB / (IRWTemp + 273.15)) - PF)) - PO
        raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind
        raw_refl2 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2
        raw_atm2 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2
        raw_obj = (raw / E / tau1 / IRT / tau2 - raw_atm1_attn -
                   raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn)

        # temperature from radiance
        temp_celcius = PB / log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15
        return temp_celcius

    @staticmethod
    def extract_float(dirtystr):
        """
        Extract the float value of a string, helpful for parsing the exiftool data
        :return:
        """
        digits = re.findall(r"[-+]?\d*\.\d+|\d+", dirtystr)
        return float(digits[0])

    def plot(self):
        """
        Plot the rgb + thermal image (easy to see the pixel values)
        :return:
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.get_thermal_np()

        plt.subplot(1, 2, 1)
        plt.imshow(thermal_np, cmap='hot')
        plt.subplot(1, 2, 2)
        plt.imshow(rgb_np)
        plt.show()

    def save_images(self):
        """
        Save the extracted images
        :return:
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.extract_thermal_image()

        img_visual = Image.fromarray(rgb_np)
        thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
        img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized) * 255))

        fn_prefix, _ = os.path.splitext(self.flir_img_filename)
        thermal_filename = fn_prefix + self.thermal_suffix
        image_filename = fn_prefix + self.image_suffix
        if self.use_thumbnail:
            image_filename = fn_prefix + self.thumbnail_suffix

        if self.is_debug:
            print("DEBUG Saving RGB image to:{}".format(image_filename))
            print("DEBUG Saving Thermal image to:{}".format(thermal_filename))

        img_visual.save(image_filename)
        img_thermal.save(thermal_filename)

    def export_thermal_to_csv(self, csv_filename):
        """
        Convert thermal data in numpy to json
        :return:
        """

        with open(csv_filename, 'w') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(['x', 'y', 'temp (c)'])

            pixel_values = []
            for e in np.ndenumerate(self.thermal_image_np):
                x, y = e[0]
                c = e[1]
                pixel_values.append([x, y, c])

            writer.writerows(pixel_values)

"""Added - PXY Nov 20"""
def cropImage(filename):
    im = Image.open(filename)
    width, height = im.size
    left = width / 6
    top = height / 2
    right = 5 * width / 6
    bottom = height
    im1 = im.crop((left, top, right, bottom))
    gray = cv2.cvtColor(np.array(im1), cv2.COLOR_BGR2GRAY)
    return width, height, gray

"""Detect circles and get coordinates of centroids"""
def getCoordinates(gray, width, height):
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    if str(gray) == str(grayrgb):
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=10, minRadius=10, maxRadius=16)
    elif str(gray) == str(graythermal):
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=10, minRadius=5, maxRadius=10)
    # print(detected_circles.shape[1])
    if detected_circles.shape[1] == 44:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        points = []
        radius = []
        ct = 1
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            points.append((a,b,ct))
            radius.append(r)
            ct = ct + 1

        """ Coordinates """
        coordinates = []
        for i in points:
            coordinates.append([i[0] + width / 6, i[1] + height / 2])

        return coordinates

"""Reconcile corresponding points"""
def sortCoordinates(coordinates):
    sortedcoor = sorted(coordinates, key=lambda k: [k[1], k[0]])
    sorted_coor = sortedcoor
    for j in range(8):
        if j%2 == 0:
            rangemin = int((j/2)*11)
            rangemax = int((j/2)*11+5)
            avg = np.mean(sortedcoor[rangemin:rangemax], axis = 0)
            for i in range(rangemin,rangemax+1,1):
                sorted_coor[i][1] = avg[1]
        else:
            rangemin = int(j*6+((j-1)/2)*(-1))
            rangemax = int(j*6+((j-1)/2)*(-1)+4)
            avg = np.mean(sortedcoor[rangemin:rangemax], axis=0)
            for i in range(rangemin,rangemax+1,1):
                sorted_coor[i][1] = avg[1]

    sorted_coor = sorted(sorted_coor, key=lambda k: [k[1], k[0]])
    return sorted_coor

def cameraCalibration(obj_points, img_points, gray):
    obj_points = np.array([obj_points], dtype = np.float32)
    img_points = np.array([img_points], dtype = np.float32)
    ret, camera_mat, distortion, rotation_vecs, translation_vecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return camera_mat, distortion, rotation_vecs, translation_vecs

def transferMatrices(obj_points, img_points, gray):
    for pt in obj_points:
        pt.append(0)
    obj_points = np.array([obj_points], dtype = np.float32)
    img_points = np.array([img_points], dtype = np.float32)
    ret, camera_mat, distortion, rotation_vecs, translation_vecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    testrot = np.array(rotation)
    testrot = testrot.astype(np.float32)
    [rot_matrix, jacobian] = cv2.Rodrigues(testrot)
    return rot_matrix, translation_vecs

def camera_calibration(thermal_filename, rgb_filename):
    widththermal, heightthermal, graythermal = cropImage(thermal_filename)
    widthrgb, heightrgb, grayrgb = cropImage(rgb_filename)

    "Adding filters to remove 'noise' and other objects"
    for i in range(grayrgb.shape[0]):
        for j in range(grayrgb.shape[1]):
            if grayrgb[i][j] < 170:
                grayrgb[i][j] = 20
            else:
                grayrgb[i][j] = 170
    for i in range(graythermal.shape[0]):
        for j in range(graythermal.shape[1]):
            if graythermal[i][j] < 95:
                graythermal[i][j] = 80
            else:
                graythermal[i][j] = 150

    coordinatesrgb = getCoordinates(grayrgb, widthrgb, heightrgb)
    coordinatesthermal = getCoordinates(graythermal, widththermal, heightthermal)

    """Object points"""
    diameter = 1.5 # cm
    c_c = 1.5+2.3 # centre-centre (vertical and horizontal)
    dist = 0 # planar calibration points
    objpoints = []
    for ii in range(8): # number of rows - specific to calibration target
        ypt = 0.5*diameter + (0.5 * (c_c) + 0.5 * diameter)*ii
        if ii%2 == 0:
            for jj in range(6):
                xpt = jj * (c_c + diameter) + 0.5 * diameter
                objpoints.append([xpt, ypt, dist])
        else:
            for jj in range(5):
                xpt = jj * (c_c + diameter) + (diameter + 0.5*c_c)
                objpoints.append([xpt, ypt, dist])

    rgb_points_unscaled = sortCoordinates(coordinatesrgb)
    thermal_points = sortCoordinates(coordinatesthermal)
    obj_points = sortCoordinates(objpoints)

    """Scaling RGB to thermal"""
    widthratio = widththermal / widthrgb
    heightratio = heightthermal / heightrgb

    rgb_points = copy.deepcopy(rgb_points_unscaled)
    for i in range(len(rgb_points)):
        rgb_points[i][0] = rgb_points_unscaled[i][0] * widthratio
        rgb_points[i][1] = rgb_points_unscaled[i][1] * heightratio

    camera_rgb, distortion_rgb, rotation_rgb, translation_rgb = cameraCalibration(obj_points, rgb_points, grayrgb)
    camera_thermal, distortion_thermal, rotation_thermal, translation_thermal = cameraCalibration(obj_points, thermal_points, graythermal)
    thermal_points_3D = copy.deepcopy(thermal_points)
    rotation, translation = transferMatrices(thermal_points_3D, rgb_points, grayrgb)
    return camera_rgb, camera_thermal, rotation, translation


def raw_nps_from_flir(img_path, verbose=False):
    fie = FlirImageExtractor(exiftool_path='exiftool', is_debug=verbose)
    fie.process_image(img_path)
    if verbose:
        fie.plot()

    rgb_np = fie.get_rgb_np()
    thermal_np = fie.get_thermal_np()
    return rgb_np, thermal_np


def extract_raws_from_dir(in_path, out_path=None):
    if out_path is None:
        out_path = f'{in_path}_raw'
    rgb_dir = os.path.join(out_path, 'rgb')
    thermal_dir = os.path.join(out_path, 'thermal')
    for path in (thermal_dir, rgb_dir):
        if not os.path.exists(path):
            os.makedirs(path)

    for i, f in enumerate(os.listdir(in_path)):
        with status(f"[bold yellow]Extracting raw RGB/T from image {i}."):
            path = os.path.join(in_path, f)
            basename = os.path.splitext(f)[0]
            rgb_np, thermal_np = raw_nps_from_flir(path, verbose=False)

            img_visual = Image.fromarray(rgb_np)

            # HACK: For now, upsample thermal image to get image of the same dimension as RGB.
            # TODO: Change NeRF model to handle images of different camera resolutions.
            h, w, _ = rgb_np.shape
            thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
            thermal_normalized = skimage.transform.resize(thermal_normalized, (h, w))
            # img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized) * 255))
            img_thermal = Image.fromarray(np.uint8(thermal_normalized * 255))

            img_visual_path = os.path.join(rgb_dir, f'{basename}_rgb.png')
            img_thermal_path = os.path.join(thermal_dir, f'{basename}_thermal.png')
            img_visual.save(img_visual_path)
            img_thermal.save(img_thermal_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and visualize Flir Image data')
    parser.add_argument('-i', '--input', type=str, help='Input image. Ex. img.jpg', required=True)
    parser.add_argument('-p', '--plot', help='Generate a plot using matplotlib', required=False, action='store_true')
    parser.add_argument('-exif', '--exiftool', type=str, help='Custom path to exiftool', required=False,
                        default='exiftool')
    parser.add_argument('-csv', '--extractcsv', help='Export the thermal data per pixel encoded as csv file',
                        required=False)
    parser.add_argument('-d', '--debug', help='Set the debug flag', required=False,
                        action='store_true')
    args = parser.parse_args()

    fie = FlirImageExtractor(exiftool_path=args.exiftool, is_debug=args.debug)
    fie.process_image(args.input)

    if args.plot:
        fie.plot()

    if args.extractcsv:
        fie.export_thermal_to_csv(args.extractcsv)

    fie.save_images()
