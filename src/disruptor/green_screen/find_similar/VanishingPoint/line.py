import math

def avg(list):
    return sum(list)/len(list)

class Line:
    def __init__(self, coords=list):
        self.x1 = coords[0]
        self.y1 = coords[1]
        self.x2 = coords[2]
        self.y2 = coords[3]

        if self.x2 == self.x1:
            raise Exception("Slope for line ", coords, " cannot be calculated, because it is vertical")

        self.m = (self.y2 - self.y1) / (self.x2 - self.x1)
        self.b = self.y1 - self.m * self.x1

    @staticmethod
    def get_averaged_lines(lines):
        groups = Line.group_lines(lines)
        print(groups)
        avg_lines = []
        for group in groups:
            slopes = [line.m for line in group]
            intercepts = [line.b for line in group]
            avg_slope = avg(slopes)
            avg_intercept = avg(intercepts)

            avg_group = []
            for line_piece in group:
                try:
                    avg_line_piece = line_piece.changed_slope(avg_slope)
                    avg_group.append(avg_line_piece)
                except Exception as e:
                    print(e)

            lowest_point = Line.get_lowest_point(avg_group)
            highest_point = Line.get_highest_point(avg_group)
            if avg_slope > 0:
                coords = [*lowest_point, *highest_point]
                avg_line = Line(coords)
                avg_lines.append(avg_line)
            elif avg_slope < 0:
                coords = [*highest_point, *lowest_point]
                avg_line = Line(coords)
                avg_lines.append(avg_line)
            else:
                continue

        return avg_lines

    @staticmethod
    def get_lowest_point(lines):
        lowest_point = [None, None]
        for line in lines:
            x1, y1, x2, y2, m, b = line.get_coords()
            if not all(lowest_point): # If lowest point is None
                if y1 < y2:
                    lowest_point = [x1, y1]
                elif y2 < y1:
                    lowest_point = [x2, y2]
            else: # If lowest point have been assigned already, we have to check if the new point is lower
                if y1 < y2:
                    if y1 < lowest_point[1]:
                        lowest_point = [x1, y1]
                elif y2 < y1:
                    if y2 < lowest_point[1]:
                        lowest_point = [x2, y2]
        return lowest_point

    @staticmethod
    def get_highest_point(lines):
        highest_point = [None, None]
        for line in lines:
            x1, y1, x2, y2, m, b = line.get_coords()
            if not all(highest_point):  # If highest point is None
                if y1 > y2:
                    highest_point = [x1, y1]
                elif y2 > y1:
                    highest_point = [x2, y2]
            else:  # If highest point have been assigned already, we have to check if the new point is higher
                if y1 > y2:
                    if y1 > highest_point[1]:
                        highest_point = [x1, y1]
                elif y2 > y1:
                    if y2 > highest_point[1]:
                        highest_point = [x2, y2]
        return highest_point

    def is_similar_to(self, line, slope_error=0.1, intercept_error=20):
        slope_diff = abs(self.m - line.m)
        intercept_diff = abs(self.b - line.b)
        if slope_diff <= slope_error and intercept_diff <= intercept_error:
            return True
        return False

    @staticmethod
    def group_lines(lines):
        groups = []
        for line in lines:
            group_found = False
            for group in groups:
                for candidate_line in group:
                    if line.is_similar_to(candidate_line):
                        group.append(line)
                        group_found = True
                        break
                if group_found:
                    break
            if not group_found:
                groups.append([line])
        return groups

    def rotated_around(self, point, angle):
        x, y = point
        relative_coords = [
            self.x1 - x,
            self.y1 - y,
            self.x2 - x,
            self.y2 - y
        ]

        # We apply rotation matrix [[cosPheta, -sinPheta], [sinPheta, cosPheta]]
        x1, y1, x2, y2 = relative_coords
        cosPheta = math.cos(angle)
        sinPheta = math.sin(angle)
        new_relative_coords = [
            x1 * cosPheta - y1 * sinPheta,
            x1 * sinPheta + y1 * cosPheta,
            x2 * cosPheta - y2 * sinPheta,
            x2 * sinPheta + y2 * cosPheta,
        ]

        x1, y1, x2, y2 = new_relative_coords
        new_coords = [
            x1 + x,
            y1 + y,
            x2 + x,
            y2 + y,
        ]

        return Line(new_coords)

    def changed_slope(self, m):
        current_angle = math.atan(self.m)
        new_angle = math.atan(m)
        diff_angle = new_angle - current_angle
        middle_point = self.get_middle_point()
        rotated_line = self.rotated_around(middle_point, diff_angle)
        return rotated_line

    def get_middle_point(self):
        x = (self.x1 + self.x2) // 2
        y = (self.y1 + self.y2) // 2
        return (x, y)

    def get_coords(self):
        return [self.x1, self.y1, self.x2, self.y2, self.m, self.b]

    def __str__(self):
        return [self.x1, self.y1, self.x2, self.y2] + " "

    def convert_into_right_format(self): # We do it for our FilterLines function which takes [[x1,y1,x2,y2], m, b] format
        return [[int(self.x1), int(self.y1), int(self.x2), int(self.y2)]]