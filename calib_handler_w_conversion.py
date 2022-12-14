from math import sqrt, degrees, acos, asin, cos, sin, radians

from scipy.optimize import minimize

class CalibrationSolverWithConversion:
    def __init__(self, target_point_list, actual_point_list, arm_values, cam_to_head_length):
        # The allowed tolerance in which values may be increased/decreased.
        self.BOUNDARY_TOLERANCE = 30

        # List of ideal points to compare.
        self.TARGET_POINT_LIST = target_point_list

        # List of real world points to compare against the target points.
        self.ACTUAL_POINT_LIST = actual_point_list

        # Number of points that will be compared.
        self.NUMBER_OF_POINTS = len(self.TARGET_POINT_LIST)

        # Proximal arm length in mm.
        self.INITIAL_PROXIMAL_LENGTH = arm_values[0]

        # Distal arm length in mm.
        self.INITIAL_DISTAL_LENGTH = arm_values[1]

        # Position of the proximal endstop in degrees.
        self.INITIAL_PROXIMAL_ANGLE = arm_values[2]

        # Position of the distal endstop in degrees.
        self.INITIAL_DISTAL_ANGLE = arm_values[3]

        # X position of the proximal joint in mm.
        self.INITIAL_PROXIMAL_JOINT_X_POSITION = arm_values[4]

        # Y position of the proximal joint in mm.
        self.INITIAL_PROXIMAL_JOINT_Y_POSITION = arm_values[5]

        # Length from the camera to the nozzle head
        self.CAM_TO_HEAD_LENGTH = cam_to_head_length

        self.boundaries = None
        self.initial_guess = None
        self.initial_error = None
        self.x_error = None
        self.y_error = None
        self.final_error = None
        self.iterations = None
        self.proximal_offset_list = None
        self.distal_offset_list = None

    def _prepare_solver(self):
        """
        Sets necessary variables to default/appropriate values.

        Parameters:
            None

        Returns:
            Nothing
        """
        self.JOINT_TO_CAM_LENGTH = self.INITIAL_DISTAL_LENGTH - self.CAM_TO_HEAD_LENGTH

        self.boundaries = ([self.INITIAL_PROXIMAL_LENGTH - self.BOUNDARY_TOLERANCE,
                            self.INITIAL_PROXIMAL_LENGTH + self.BOUNDARY_TOLERANCE],
                           [self.JOINT_TO_CAM_LENGTH - self.BOUNDARY_TOLERANCE,
                            self.JOINT_TO_CAM_LENGTH + self.BOUNDARY_TOLERANCE],
                           [self.CAM_TO_HEAD_LENGTH - self.BOUNDARY_TOLERANCE,
                            self.CAM_TO_HEAD_LENGTH + self.BOUNDARY_TOLERANCE],
                           [self.INITIAL_PROXIMAL_ANGLE - self.BOUNDARY_TOLERANCE,
                            self.INITIAL_PROXIMAL_ANGLE + self.BOUNDARY_TOLERANCE],
                           [self.INITIAL_DISTAL_ANGLE - self.BOUNDARY_TOLERANCE,
                            self.INITIAL_DISTAL_ANGLE + self.BOUNDARY_TOLERANCE],
                           [self.INITIAL_PROXIMAL_JOINT_X_POSITION - self.BOUNDARY_TOLERANCE,
                            self.INITIAL_PROXIMAL_JOINT_X_POSITION + self.BOUNDARY_TOLERANCE],
                           [self.INITIAL_PROXIMAL_JOINT_Y_POSITION - self.BOUNDARY_TOLERANCE,
                            self.INITIAL_PROXIMAL_JOINT_Y_POSITION + self.BOUNDARY_TOLERANCE])

        self.initial_guess = [self.INITIAL_PROXIMAL_LENGTH, self.JOINT_TO_CAM_LENGTH, self.CAM_TO_HEAD_LENGTH,
                              self.INITIAL_PROXIMAL_ANGLE, self.INITIAL_DISTAL_ANGLE,
                              self.INITIAL_PROXIMAL_JOINT_X_POSITION, self.INITIAL_PROXIMAL_JOINT_Y_POSITION]

        self.initial_error = 0
        self.x_error = [[0] for _ in range(self.NUMBER_OF_POINTS)]
        self.y_error = [[0] for _ in range(self.NUMBER_OF_POINTS)]
        self.final_error = 0
        self.iterations = 0

    def _cumulative_error(self, params):
        """
        Calculates the cumulative error for the points in the list.

        Parameters:
            params (list): List of parameters. ie. [final_proximal_length, joint_to_cam, cam_to_head_length,
                                                    final_proximal_angle, final_distal_angle,
                                                    final_proximal_joint_x_position, final_proximal_joint_y_position]

        Returns:
            Nothing
        """
        error = 0
        x_error_sum = 0
        y_error_sum = 0
        self.iterations += 1
        for point in range(0, self.NUMBER_OF_POINTS):
            x_target, y_target = self.calc_head_from_cam(self.TARGET_POINT_LIST[point][0], self.TARGET_POINT_LIST[point][1], params)
            x_actual, y_actual = self.ACTUAL_POINT_LIST[point][0], self.ACTUAL_POINT_LIST[point][1]

            self.x_error[point] = x_target - x_actual
            self.y_error[point] = y_target - y_actual

            x_error_sum += self.x_error[point] ** 2
            y_error_sum += self.y_error[point] ** 2

        error = sqrt(x_error_sum + y_error_sum)
        self.final_error = error
        return error

    def calc_head_from_cam(self, xc: float, yc: float, params: list, debug=False):
        """
        Calculates the position of the print head from the position of the camera
        Parameters
            xc: recorded x position of camera
            yc: recorded y position of camera
            params: list of configuration with distal length split into joint_to_cam and cam_to_head

        Returns
            xn: calculated x position of head
            yn: calculated y position of head
        """
        if debug:
            print(f"params: {params}")
            print(f"(xc, yc): ({xc}, {yc})")
        proximal_length = params[0]
        # Note: joint_to_cam_length + cam_to_head_length = distal_length
        joint_to_cam_length = params[1]
        cam_to_head_length = params[2]
        base_x = params[5]
        base_y = params[6]

        # c is the length of the third side of triangle defined by proximal and distal arm
        c = sqrt((base_x - xc) ** 2 + (yc - base_y) ** 2)
        # B is the angle opposite of the distal arm
        B = degrees(acos((proximal_length ** 2 + c ** 2 - (joint_to_cam_length ** 2)) / (2 * proximal_length * c)))
        # C is the angle opposite of the third side of the triangle defined by the proximal and distal arm
        C = degrees(acos((proximal_length ** 2 + joint_to_cam_length ** 2 - c ** 2) / (2 * proximal_length * joint_to_cam_length)))

        # Calculate intermediate angle values
        if xc < base_x:
            P = 180 - B - degrees(asin((yc - base_y) / c))
        elif xc >= base_x:
            P = degrees(asin((yc - base_y) / c)) - B
        D = 180 - C

        # Calculate where the print head is when the camera is at (xc, yc)
        xn = xc - cam_to_head_length * cos(radians(180 - P - D))
        yn = yc + cam_to_head_length * sin(radians(180 - P - D))

        return xn, yn

    def execute_calibration(self):
        """
        Executes the printer calibration.

        Parameters:
            None

        Returns:
            gcode (str): Printer configuration G-Code.
            final_error (float): The final error calculate for running.
        """
        # ACTUAL_POINT_LIST = arm_point_list or the ground truth coords
        print('Actual point list: {}'.format(self.ACTUAL_POINT_LIST))
        # TARGET_POINT_LIST = calibration_point_list or recorded camera vals
        print('Target point list: {}'.format(self.TARGET_POINT_LIST))

        self._prepare_solver()

        print("Calculating...")

        initial_error = self._cumulative_error(self.initial_guess)
        result = minimize(self._cumulative_error, self.initial_guess,
                          method='trust-constr', bounds=self.boundaries, options={'maxiter': 10000, 'verbose': 0, })

        if result.success:
            print("Initial error: {:4.3f}mm".format(initial_error))

            for point in range(0, self.NUMBER_OF_POINTS):
                print("point {} error:\t{:7.3f}, {:7.3f}".format(point + 1, self.x_error[point], self.y_error[point]))

            print("Final error: {:7.4f}mm".format(self.final_error))
            gcode = "M669 K4 P{:.5f} D{:.5f} A{:.5f}:150 B20:{:.5f} X{:.5f} Y{:.5f}".format(
                result.x[0], (result.x[1] + result.x[2]), result.x[3], result.x[4], -result.x[5], -result.x[6])

            # result.x[2] is the new cam_to_head_length
            return gcode, result.x[2], self.final_error
        else:
            raise ValueError(result.message)

if __name__ == '__main__':
    # calibration_point will be the points that the camera finds when the nozzle is instructed to go to the arm_point
    calibration_point_list = [(115, 87), (138, 195), (518, 94), (505, -55)]
    # arm_point will hold the nozzle head values that we are calibrating to
    arm_point_list = [(0, 0), (0, 250), (550, 250), (550, 100), (150, 100)]
    # arm_values will be the initial config values that are used to retrieve calibration_point
    arm_vals = [217.93967, 255.27140, -49.3681, 170.43576, 283.20797, -52.23731]
    # length from camera to nozzle head
    # currently separated for arm values for testing purposes
    cam_head_length = 115

    calibrationSolver = CalibrationSolverWithConversion(calibration_point_list, arm_point_list, arm_vals, cam_head_length)
    configuration_gcode, final_cam_to_head_length, calibration_error = calibrationSolver.execute_calibration()
    print(f"Final configuration code: {configuration_gcode} \nFinal cam_to_head_length: {final_cam_to_head_length}")
