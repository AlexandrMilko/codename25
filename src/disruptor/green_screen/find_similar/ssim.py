# USAGE
# python image_diff.py --first images/original_01.png --second images/modified_01.png

# import the necessary packages
from disruptor.green_screen.find_similar.VanishingPoint.main import ReadImage
from disruptor.green_screen.find_similar.VanishingPoint.main import GetLines
from disruptor.green_screen.find_similar.VanishingPoint.main import GetVanishingPoint
from disruptor.green_screen.find_similar.VanishingPoint.main import manhattan_distance

from skimage.metrics import structural_similarity as compare_ssim
import argparse
import cv2
from PIL import Image
import numpy as np

def convert_to_cv2(pil_image):
	pil_image = pil_image.convert("RGB")
	open_cv_image = np.array(pil_image)
	# Convert RGB to BGR
	return open_cv_image[:, :, ::-1].copy()

def convert_to_pil(cv2_image):
	img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
	return Image.fromarray(img)

def round_contours(cv2_image):
	# find the mask
	img_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
	# set some absolute values for the threshold, since we know the background will always be white
	_, mask = cv2.threshold(img_gray, 244, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)

	# find the largest contour
	contours, _ = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

	# approximate the contour with a curve
	epsilon = 0.015 * cv2.arcLength(largest_contour, True)
	largest_contour_approx = cv2.approxPolyDP(largest_contour, epsilon, True)

	# draw the largest contour to fill in the holes in the mask
	final_result = np.ones(cv2_image.shape[:2])  # create a blank canvas to draw the final result
	final_result = cv2.drawContours(final_result, [largest_contour_approx], -1, color=(0, 255, 0), thickness=cv2.FILLED)

	# Draw the red border around the largest contour
	final_result_with_border = cv2.drawContours(cv2_image.copy(), [largest_contour], -1, color=(0, 0, 255), thickness=2)
	# Show results
	# cv2.imshow('final_result_with_border', final_result_with_border)

	# show results
	# cv2.imshow('mask', mask)

	# Define color mapping
	color_mapping = {
		0: (0, 0, 0),  # Black
		1: (255, 255, 255)  # White
	}

	# Convert binary image to BGR with custom color mapping
	bgr_image = np.zeros((final_result.shape[0], final_result.shape[1], 3), dtype=np.uint8)
	for key, value in color_mapping.items():
		bgr_image[final_result == key] = value

	# cv2.imshow("bgr.jpg", bgr_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return bgr_image

def mask_everything_except(colors, image, color_error=1):
	# COLORS is a list np.arrays of colors in BGR format [[B1,G1,R1], [B2,G2,R2], ...]
	# image - cv2 image
	masked_image = image
	mask = np.zeros_like(image[:,:,0], dtype=bool)  # Initialize mask with zeros
	for color in colors:
		# Create a mask to identify the regions with the preserve_color
		mask += np.all(np.abs(masked_image - color) <= color_error, axis=-1)
	# Create a new image filled with (255, 255, 255)
	masked_image = np.full_like(masked_image, (255, 255, 255), dtype=np.uint8)
	# Copy the original image to the new image where the mask is True
	masked_image[mask] = image[mask]
	# Showing the final image
	# cv2.imshow("OutputImageMasked", masked_image)
	# cv2.waitKey(0)
	return masked_image

def point_lies_on_lines(point, line1, line2, error=15):
	x, y = point
	x1, y1, x2, y2 = line1
	x3, y3, x4, y4 = line2
	if (min(x1, x2) - error) <= x <= (max(x1, x2) + error) and (min(y1, y2) - error) <= y <= (
			max(y1, y2) + error) \
			and (min(x3, x4) - error) <= x <= (max(x3, x4) + error) and (
			min(y3, y4) - error) <= y <= (max(y3, y4) + error):
		return True
	else:
		return False

def are_lines_intersecting(line1, line2):
	img = np.zeros((1024, 1024, 3), dtype=np.uint8)  # Create a black image
	x1, y1, x2, y2 = line1[:4]
	x3, y3, x4, y4 = line2[:4]

	# Draw the lines
	cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Line 1 in green
	cv2.line(img, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Line 2 in red

	# Check if the lines are not parallel
	det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
	if det == 0:
		return False  # Lines are parallel

	# Calculate intersection point
	intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
	intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

	cv2.circle(img, (int(intersection_x), int(intersection_y)), 5, (255, 255, 255), -1)  # Intersection point in white

	# Show the image
	# cv2.imshow('Lines and Intersection', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Check if intersection point is within the line segments
	return point_lies_on_lines([intersection_x, intersection_y], [x1, y1, x2, y2], [x3, y3, x4, y4])

def remove_edges(lines): # Remove lines which connect 2 other lines.
	lines_without_edges = lines.copy()
	for line in lines:
		connecting_counter = 0
		for compare_line in lines:
			if(line != compare_line):
				# print(are_lines_intersecting(line, compare_line), " INTERSECTING: ", line, compare_line)
				if are_lines_intersecting(line, compare_line):
					connecting_counter += 1
				if connecting_counter >= 2: # It is not going to be bigger than 2, but just in case.
					lines_without_edges.remove(line)
					break
	print(len(lines), len(lines_without_edges), "LENGTHS", sep="\n")
	return lines_without_edges


def identify_room_type(lines):
	# 1. Corner - when we have only 2 walls present in image, resulting in a corner.
	# 2. Vanishing Point room - when we are able to calculate vanishing point for a room. It has to have more than 2 lines.
	# Some line will be connecting other lines and we have to ignore it to calculate vanishing point.
	# (3. 2 Lines - it comes as a result of removing horizontal line,
	# which connects the rest of lines from the image in FilterLines function. But I changed REJECT_DEGREE_TH to 0, to preserve all lines.
	# So we will never encounter such type of room. Writing it here just in case.)
	if len(lines) == 2:
		if are_lines_intersecting(lines[0], lines[1]):
			return "corner"
		else:
			return "vanishing point" # it is actually the third type, but we can compare "2 lines" and "vanishing point".
		# "2 Lines" is a specific case of "vanishing point"
	return "vanishing point"

def compare_vanishing_point(first_path, second_path):
	first_image = cv2.imread(first_path)
	second_image = cv2.imread(second_path)

	# # Fill everything with black except for floor, to calculate vanishing point only based on floor
	# # Define the color you want to preserve, BGR format
	floor_color = np.array([50, 50, 80])
	filled_first_image = mask_everything_except([floor_color], first_image)
	filled_second_image = mask_everything_except([floor_color], second_image)

	# Round the contours of floor to decrease noise
	round_filled_first_image = round_contours(filled_first_image)
	round_filled_second_image = round_contours(filled_second_image)

	# Getting the lines form the image
	first_lines = GetLines(round_filled_first_image)
	second_lines = GetLines(round_filled_second_image)
	if not all((first_lines, second_lines)):
		raise Exception("No Lines were detected on some of the images.")

	first_room_type = identify_room_type(first_lines)
	second_room_type = identify_room_type(second_lines)
	if first_room_type != second_room_type:
		raise Exception("Rooms have different types: ", first_room_type, " and ", second_room_type)

	first_lines = remove_edges(first_lines)
	second_lines = remove_edges(second_lines)

	# Get vanishing point
	first_vanishing_point = GetVanishingPoint(first_lines)
	second_vanishing_point = GetVanishingPoint(second_lines)

	# Checking if vanishing point found
	if not all((first_vanishing_point, second_vanishing_point)):
		raise Exception(
			"Vanishing Point not found. Possible reason is that not enough lines are found in the image for determination of vanishing point."
		)

	# Drawing lines and vanishing point
	for Line in first_lines:
		cv2.line(first_image, (Line[0], Line[1]), (Line[2], Line[3]), (0, 255, 0), 2)
	cv2.circle(first_image, (int(first_vanishing_point[0]), int(first_vanishing_point[1])), 10, (0, 0, 255), -1)

	# Drawing lines and vanishing point
	for Line in second_lines:
		cv2.line(second_image, (Line[0], Line[1]), (Line[2], Line[3]), (0, 255, 0), 2)
	cv2.circle(second_image, (int(second_vanishing_point[0]), int(second_vanishing_point[1])), 10, (0, 0, 255), -1)

	cv2.imshow("First Image", first_image)
	cv2.imshow("Second Image", second_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return manhattan_distance(first_vanishing_point, second_vanishing_point)

def compare_iou(first_path, second_path):
	# Open the image using PIL
	first_image = Image.open(first_path)
	second_image = Image.open(second_path)

	floor_color = np.array([50, 50, 80])
	window_color = np.array([230, 230, 230])
	door_color = np.array([51, 255, 8])
	colors = [floor_color, window_color, door_color]
	masked_first_image = convert_to_pil(mask_everything_except(colors, convert_to_cv2(first_image)))
	masked_second_image = convert_to_pil(mask_everything_except(colors, convert_to_cv2(second_image)))

	# Convert the image to a NumPy array
	first_array = np.array(masked_first_image)
	second_array = np.array(masked_second_image)

	# Calculate the intersection (common area)
	intersection = first_array & second_array

	# Calculate the union (combined area)
	union = first_array | second_array

	# Calculate IoU
	iou = float(intersection.sum()) / float(union.sum())

	return iou


# TODO rename to ssim comparison
def compare(first, second):
	# load the two input images
	imageA = cv2.imread(first)
	imageB = cv2.imread(second)

	# convert the images to grayscale
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	(score, diff) = compare_ssim(grayA, grayB, full=True)
	diff = (diff * 255).astype("uint8")
	return score

if __name__ == "__main__":
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--first", required=True,
		help="first input image")
	ap.add_argument("-s", "--second", required=True,
		help="second")
	args = vars(ap.parse_args())

	# load the two input images
	imageA = cv2.imread(args["first"])
	imageB = cv2.imread(args["second"])

	# convert the images to grayscale
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	(score, diff) = compare_ssim(grayA, grayB, full=True)
	diff = (diff * 255).astype("uint8")
	print("SSIM: {}".format(score))

	# threshold the difference image, followed by finding contours to
	# obtain the regions of the two input images that differ
	thresh = cv2.threshold(diff, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cv2.imshow("Thresh", thresh)
	cv2.waitKey(0)
