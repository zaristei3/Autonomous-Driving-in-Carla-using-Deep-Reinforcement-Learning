import time
import random
import numpy as np
import pygame
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor
from simulation.settings import *


class CarlaEnvironment():

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True) -> None:


        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.good_vehicle = None
        self.bad_vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        
        # Objects to be kept alive
        self.good_camera_obj = None
        self.bad_camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.create_pedestrians()



    # A reset function for reseting our environment.
    def reset(self):

        try:
            
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()


            # Blueprint of our main vehicle
            good_vehicle_bp = self.get_vehicle(CAR_NAME)
            bad_vehicle_bp = self.get_vehicle(CAR_NAME)

            if self.town == "Town02":
                choices = [self.map.get_spawn_points()[6], self.map.get_spawn_points()[11]]
                good_transform, bad_transform = choices[0], choices[1]
                #Town2 is 11 or 6
                self.total_distance = 780
            else:
                raise Exception("Town not supported")

            self.good_vehicle = self.world.try_spawn_actor(good_vehicle_bp, good_transform)
            self.bad_vehicle = self.world.try_spawn_actor(bad_vehicle_bp, bad_transform)
            self.actor_list.append(self.good_vehicle)
            self.actor_list.append(self.bad_vehicle)


            # Camera Sensor
            self.good_camera_obj = CameraSensor(self.good_vehicle)
            self.bad_camera_obj = CameraSensor(self.bad_vehicle)
            while(len(self.good_camera_obj.front_camera) == 0 or len(self.bad_camera_obj.front_camera) == 0):
                time.sleep(0.0001)
            self.good_image_obs = self.good_camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.good_camera_obj.sensor)

            self.bad_image_obs = self.bad_camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.bad_camera_obj.sensor)

            # Third person view of our vehicle in the Simulated env
            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.good_vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.good_vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            
            self.timesteps = 0
            self.good_rotation = self.good_vehicle.get_transform().rotation.yaw
            self.bad_rotation = self.bad_vehicle.get_transform().rotation.yaw
            self.good_previous_location = self.good_vehicle.get_location()
            self.bad_previous_location = self.bad_vehicle.get_location()
            self.center_lane_deviation = 0.0
            self.target_speed = 22 #km/h
            self.max_speed = 25.0
            self.min_speed = 15.0
            self.max_distance_from_center = 3
            self.good_throttle = float(0.0)
            self.bad_throttle = float(0.0)
            self.good_previous_steer = float(0.0)
            self.bad_previous_steer = float(0.0)
            self.good_velocity = float(0.0)
            self.bad_velocity = float(0.0)
            self.good_distance_from_center = float(0.0)
            self.bad_distance_from_center = float(0.0)
            self.good_angle = float(0.0)
            self.bad_angle = float(0.0)
            self.distance_covered = 0.0


            if self.fresh_start:
                self.current_waypoint_index = 0
                # Waypoint nearby angle and distance from it
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(self.good_vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                for x in range(self.total_distance):
                    if self.town == "Town02":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        raise Exception("Town not supported")
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
            else:
                # Teleport vehicle to last checkpoint
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.good_vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            self.good_navigation_obs = np.array([self.good_throttle, self.good_velocity, self.good_previous_steer, self.good_distance_from_center, self.good_angle])
            self.bad_navigation_obs = np.array([self.bad_throttle, self.bad_velocity, self.bad_previous_steer, self.bad_distance_from_center, self.bad_angle])

                        
            time.sleep(0.5)
            self.collision_history.clear()

            self.episode_start_time = time.time()
            return [self.good_image_obs, self.good_navigation_obs], [self.bad_image_obs, self.bad_navigation_obs]

        except Exception as e:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()
            raise e


# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent|
# ----------------------------------------------------------------

    # A step function is used for taking inputs generated by neural net.
    def step(self, action_idx):
        try:

            self.timesteps+=1
            self.fresh_start = False

            # Velocity of the vehicle
            good_velocity = self.good_vehicle.get_velocity()
            self.good_velocity = np.sqrt(good_velocity.x**2 + good_velocity.y**2 + good_velocity.z**2) * 3.6

            bad_velocity = self.bad_vehicle.get_velocity()
            self.bad_velocity = np.sqrt(bad_velocity.x**2 + bad_velocity.y**2 + bad_velocity.z**2) * 3.6
            
            good_action_idx, bad_action_idx = action_idx[0], action_idx[1]

            # Action fron action space for contolling the vehicle with a discrete action
            if self.continous_action_space:
                good_steer = float(good_action_idx[0])
                good_steer = max(min(good_steer, 1.0), -1.0)
                bad_steer = float(bad_action_idx[0])
                bad_steer = max(min(bad_steer, 1.0), -1.0)
                
                good_throttle = float((good_action_idx[1] + 1.0)/2)
                good_throttle = max(min(good_throttle, 1.0), 0.0)
                bad_throttle = float((bad_action_idx[1] + 1.0)/2)
                bad_throttle = max(min(bad_throttle, 1.0), 0.0)

                self.good_vehicle.apply_control(carla.VehicleControl(steer=self.good_previous_steer*0.9 + good_steer*0.1, throttle=self.good_throttle*0.9 + good_throttle*0.1))
                self.bad_vehicle.apply_control(carla.VehicleControl(steer=self.bad_previous_steer*0.9 + bad_steer*0.1, throttle=self.bad_throttle*0.9 + bad_throttle*0.1))
                self.good_previous_steer = good_steer
                self.bad_previous_steer = bad_steer
                self.good_throttle = good_throttle
                self.bad_throttle = bad_throttle
            else:
                good_steer = self.good_action_space[good_action_idx]
                bad_steer = self.bad_action_space[bad_action_idx]
                if self.good_velocity < 20.0:
                    self.good_vehicle.apply_control(carla.VehicleControl(steer=self.good_previous_steer*0.9 + good_steer*0.1, throttle=1.0))
                else:
                    self.good_vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + good_steer*0.1))
                
                if self.bad_velocity < 20.0:
                    self.bad_vehicle.apply_control(carla.VehicleControl(steer=self.bad_previous_steer*0.9 + bad_steer*0.1, throttle=1.0))
                else:
                    self.bad_vehicle.apply_control(carla.VehicleControl(steer=self.bad_previous_steer*0.9 + bad_steer*0.1))

                self.good_previous_steer = good_steer
                self.bad_previous_steer = bad_steer
                self.good_throttle = 1.0
                self.bad_throttle = 1.0

            
            # Traffic Light state
            if self.good_vehicle.is_at_traffic_light():
                traffic_light = self.good_vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            self.collision_history = self.collision_obj.collision_data            

            # Rotation of the vehicle in correlation to the map/lane
            self.good_rotation = self.good_vehicle.get_transform().rotation.yaw
            self.bad_rotation = self.bad_vehicle.get_transform().rotation.yaw

            # Location of the car
            self.good_location = self.good_vehicle.get_location()
            self.bad_location = self.bad_vehicle.get_location()

            #transform = self.vehicle.get_transform()
            # Keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # Check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.good_location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break
            
            self.current_waypoint_index = waypoint_index
            # Calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.good_distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.good_location))
            self.center_lane_deviation += self.good_distance_from_center


            # Get angle difference between closest waypoint and vehicle forward vector
            fwd    = self.vector(self.good_vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.good_angle  = self.angle_diff(fwd, wp_fwd)

             # Update checkpoint for training
            if not self.fresh_start:
                if self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

            
            # Rewards are given below!
            done = False
            reward = 0

            if len(self.collision_history) != 0:
                done = True
                reward = -10
            elif self.good_distance_from_center > self.max_distance_from_center:
                done = True
                reward = -10
            elif self.episode_start_time + 10 < time.time() and self.good_velocity < 1.0:
                reward = -10
                done = True
            elif self.good_velocity > self.max_speed:
                reward = -10
                done = True

            # Interpolated from 1 when centered to 0 when 3 m from center
            centering_factor = max(1.0 - self.good_distance_from_center / self.max_distance_from_center, 0.0)
            # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
            angle_factor = max(1.0 - abs(self.good_angle / np.deg2rad(20)), 0.0)

            if not done:
                if self.continous_action_space:
                    if self.good_velocity < self.min_speed:
                        reward = (self.good_velocity / self.min_speed) * centering_factor * angle_factor    
                    elif self.good_velocity > self.target_speed:               
                        reward = (1.0 - (self.good_velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor  
                    else:                                         
                        reward = 1.0 * centering_factor * angle_factor 
                else:
                    reward = 1.0 * centering_factor * angle_factor

            if self.timesteps >= 7500:
                done = True
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance//2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0

            while(len(self.good_camera_obj.front_camera) == 0 or len(self.bad_camera_obj.front_camera) == 0):
                time.sleep(0.0001)

            self.good_image_obs = self.good_camera_obj.front_camera.pop(-1)
            good_normalized_velocity = self.good_velocity/self.target_speed
            normalized_distance_from_center = self.good_distance_from_center / self.max_distance_from_center
            good_normalized_angle = abs(self.good_angle / np.deg2rad(20))
            self.good_navigation_obs = np.array([self.good_throttle, self.good_velocity, good_normalized_velocity, normalized_distance_from_center, good_normalized_angle])

            self.bad_image_obs = self.bad_camera_obj.front_camera.pop(-1)
            bad_normalized_velocity = self.bad_velocity/self.target_speed
            self.bad_navigation_obs = np.array([self.bad_throttle, self.bad_velocity, bad_normalized_velocity, self.bad_distance_from_center, self.bad_angle])
            
            # Remove everything that has been spawned in the env
            if done:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                
                for sensor in self.sensor_list:
                    sensor.destroy()
                
                self.remove_sensors()
                
                for actor in self.actor_list:
                    actor.destroy()
            
            return [self.good_image_obs, self.good_navigation_obs], [self.bad_image_obs, self.bad_navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation]

        except Exception as e:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()
            raise e



# -------------------------------------------------
# Creating and Spawning Pedestrians in our world |
# -------------------------------------------------

    # Walkers are to be included in the simulation yet!
    def create_pedestrians(self):
        try:

            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            #self.world.set_pedestrians_cross_factor(0.0)
            # 3. Starting the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except Exception as e:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])
            raise e


# ---------------------------------------------------
# Creating and Spawning other vehciles in our world|
# ---------------------------------------------------


    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawing them into the world
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except Exception as e:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])
            raise e


# ----------------------------------------------------------------
# Extra very important methods: their names explain their purpose|
# ----------------------------------------------------------------

    # Setter for changing the town on the server.
    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)


    # Getter for fetching the current state of the world that simulator is in.
    def get_world(self) -> object:
        return self.world


    # Getter for fetching blueprint library of the simulator.
    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()


    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle


    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])


    def get_discrete_action_space(self):
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint


    # Spawn the vehicle in the environment
    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)


    # Clean up method
    def remove_sensors(self):
        self.good_camera_obj = None
        self.bad_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None

