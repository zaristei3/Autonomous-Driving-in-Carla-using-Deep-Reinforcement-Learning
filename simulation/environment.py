import math
import time
import random
import numpy as np
import logging
import pygame
from simulation.connection import carla, logging
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor, LaneInvasionSensor
from simulation.settings import *


class CarlaEnvironment():

    def __init__(self, client, world) -> None:

        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = CONTINUOUS_ACTION
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.from_start=True
        self.new_location_wp = None
        #self.route_waypoints = list()
        
        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.create_pedestrians()
        logging.info("CarlaEnvironment obj has been initialized!")


    # A reset function for reseting our environment.
    def reset(self):

        try:
            
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()

            # Spawn points of the entire map!
            #spawn_points = self.map.get_spawn_points()

            # Blueprint of our main vehicle
            vehicle_bp = self.get_vehicle(CAR_NAME)

            #vehicle = self.actor_vehicle(vehicle_bp, spawn_points)
            #self.set_vehicle(vehicle_bp, spawn_points)
            transform = self.map.get_spawn_points()[38]
            transform.location += carla.Location(z=1.0)
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            self.actor_list.append(self.vehicle)


            # Camera Sensor
            self.camera_obj = CameraSensor(self.vehicle)
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)

            # Third person view of our vehicle in the Simulated env
            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)


            # Lane Invasion sensor
            self.lane_invasion_obj = LaneInvasionSensor(self.vehicle)
            self.wrong_maneuver = self.lane_invasion_obj.wrong_maneuver
            self.sensor_list.append(self.lane_invasion_obj.sensor)

            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 20 #km/h
            self.max_speed = 22.0
            self.min_speed = 15.0
            self.max_distance_from_center = 3.0
            self.total_distance = 2000
            self.throttle = float(0.0000000)
            self.previous_steer = float(0.0000000)
            self.velocity = float(0.0000000)
            self.distance_from_center = float(0.000000)
            self.angle = float(0.000000)

            if not self.from_start:
            # Teleport vehicle to last checkpoint
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                transform.location += carla.Location(z=1.0)
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            else:
                if self.new_location_wp is not None:
                    transform = self.new_location_wp.transform
                    transform.location += carla.Location(z=1.0)
                    self.vehicle.set_transform(transform)

                self.current_waypoint_index = 0
                # Waypoint nearby angle and distance from it
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                for _ in range(self.total_distance):
                    next_waypoint = current_waypoint.next(1.0)[0]
                    #if len(wp_list) > 2 and wp_list[0].is_junction:
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
                #transform = self.waypoint.transform
                #transform.location += carla.Location(z=1.0)
                #self.vehicle.set_transform(transform)
            #self.start_waypoint_index = self.current_waypoint_index
            self.navigation_obs = np.array([self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle])

            logging.info("Environment has been resetted.")
            
            self.episode_start_time = time.time()
            self.collision_history.clear()
            
            time.sleep(0.5)
            
            return [self.image_obs, self.navigation_obs]

        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()


# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent|
# ----------------------------------------------------------------

    # A step function is used for taking inputs generated by neural net.


    def step(self, action_idx):
        try:

            if self.current_waypoint_index >= 2000:
                self.new_location_wp = self.route_waypoints[-1]
                self.from_start = True
            else:
                self.from_start = False

            self.timesteps+=1
            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

            # Action fron action space for contolling the vehicle with a discrete action
            if self.continous_action_space:
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                throttle = float((action_idx[1] + 1.0)/2)
                throttle = max(min(throttle, 1.0), 0.0)
                self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle))
                self.previous_steer = steer
                self.throttle = throttle
            else:
                action = self.action_space[action_idx]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(carla.VehicleControl(steer=action, throttle=float(0.5)))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=action))
                self.previous_steer = action
        
            # Traffic Light state
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)



            self.collision_history = self.collision_obj.collision_data
            self.wrong_maneuver = self.lane_invasion_obj.wrong_maneuver
            

            # Rotation of the vehicle in correlation to the map/lane
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # Location of the car
            self.location = self.vehicle.get_location()


            #transform = self.vehicle.get_transform()
            # Keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # Check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            # Calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            #self.center_lane_deviation += self.distance_from_center

            # Get angle difference between closest waypoint and vehicle forward vector
            fwd    = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle  = self.angle_diff(fwd, wp_fwd)
            

            self.distance_traveled += (math.sqrt((self.previous_location.x - self.location.x)**2 + (self.previous_location.y - self.location.y)**2))
            #self.previous_location.distance(self.location)
            self.previous_location = self.location

            self.desired_distance = self.current_waypoint_index // len(self.route_waypoints)
            if self.desired_distance >= 1:
                done = True
                    
            # Update checkpoint for training
            if not self.from_start:
                checkpoint_frequency = 50
                self.checkpoint_waypoint_index = (self.current_waypoint_index // checkpoint_frequency) * checkpoint_frequency
            
            # Rewards are given below!
            done = False
            reward = 0

            if len(self.collision_history) != 0:
                done = True
                reward = -10
                logging.warning("Vehicle has collided.")
            elif self.distance_from_center > self.max_distance_from_center:
                done = True
                reward = -10
                logging.warning("Vehicle has gone out of the lane.")
            elif self.episode_start_time + 5 < time.time() and self.velocity < 1.0:
                logging.warning("Vehicle has stopped moving.")
                reward = -10
                done = True
            elif self.velocity > self.max_speed:
                logging.warning("Vehicle is moving too fast.")
                reward = -10
                done = True
            

            if not done:
                # Interpolated from 1 when centered to 0 when 3 m from center
                centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
                # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
                angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

                if self.velocity < self.min_speed:
                    reward = (self.velocity / self.min_speed) * centering_factor * angle_factor    
                elif self.velocity > self.target_speed:               
                    reward = (1.0 - (self.velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor  
                else:                                         
                    reward = 1.0 * centering_factor * angle_factor                 

            if self.timesteps >= 10000:
                logging.warning("10000 steps run complete.")
                done = True

            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)

            self.navigation_obs = np.array([self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle])
            # Remove everything that has been spawned in the env
            if done:
                for sensor in self.sensor_list:
                    sensor.destroy()
                self.remove_sensors()
                for actor in self.actor_list:
                    actor.destroy()
            return [self.image_obs, self.navigation_obs], reward, done, None

        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()



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

            # 3. Starting the motion of our pedestrians
            # set how many pedestrians can cross the road
            self.world.set_pedestrians_cross_factor(0)
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

            logging.info("NPC pedestrians / walkers have been generated.")
        except:
            logging.info(
                "Unfortunately, we couldn't create pedestrians in our world.")
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])


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
        except:
            logging.warning(
                "Unfortunately, we couldn't create other ai vehicles in our world.")
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])


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
             0.50])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name) -> object:
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
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None
        logging.debug("All the sensors have been removed.")
