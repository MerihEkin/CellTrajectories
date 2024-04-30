import os
import sys
import csv
import cv2
import pims
import pickle
import trackpy
import imageio
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from circle_fit import taubinSVD
from scipy.spatial import ConvexHull
import matplotlib.patches as patches

import btrack
from btrack import btypes

sys.path.append(".")

import video_processor_utils as utils

trackpy.quiet(suppress=True)


class VideoProcessor:
    """
    Video processing functions to obtain cell trajectories 
    from fark-field microscopy videos. 

    Combines trackpy detection with btrack tracking
    best of each libraries for our data.
    """

    def __init__(self) -> None:
        pass


    def bg_sliding_window(self, input_video_path, output_video_path, window_size = 300, fps=30, filter=False):
        """
        input_video_path : raw video file location
        output_video_path : background transparent video file path
        window_size : memory for background removal window
        fps : video frame per second
        
        takes input file name and writes background transparent video
        to location output_file_name
        """

        input_video = utils.gray(pims.Video(input_video_path))

        number_of_frames = len(input_video)

        with imageio.get_writer(output_video_path, fps=fps) as writer:
            for n in range(0, number_of_frames - window_size, window_size):
                bgd = np.median(input_video[n:n+window_size], axis=0)
                for i in range(n, n+window_size):
                    frame = np.abs(input_video[i] - bgd)
                    if filter:
                        frame = utils.spatialFiltering(frame)
                    writer.append_data(np.uint8(frame))

            bgd = np.median(input_video[n:], axis=0)
            for i in range(n+window_size, number_of_frames):
                frame = np.abs(input_video[i] - bgd)
                if filter:
                    frame = utils.spatialFiltering(frame)
                writer.append_data(np.uint8(frame))


    def memory_efficient_bg_removal(self, input_video_path, output_video_path, window_size = 300, filter=False):
        """
        input_video_path : raw video file location
        output_video_path : background transparent video file path
        window_size : memory for background removal window
        
        takes input file name and writes background transparent video
        to location output_file_name, uses less memory
        """
        reader = imageio.get_reader(input_video_path)
        fps = reader.get_meta_data()['fps']

        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=window_size)

        writer = imageio.get_writer(output_video_path, fps=fps)

        for frame in tqdm(reader):
            fg_mask = bg_subtractor.apply(frame)
            result = cv2.bitwise_and(frame, frame, mask=fg_mask)
            if filter:
                result = utils.spatialFiltering(result)
            writer.append_data(result)

        writer.close()
        reader.close()


    def bg_subtraction_downsampling(self, input_video_path, output_video_path, downsampling_factor = 100, fps=30, filter=False):
        """
        input_video_path : raw video file location
        output_video_path : background transparent video file path
        downsampling_factor : downsample raw video by this factor and calculate bg
        using the median frames 
        
        takes input file name and writes background transparent video
        to location output_file_name
        """
        fps = fps

        input_video = utils.gray(pims.Video(input_video_path))

        number_of_frames = len(input_video)

        assert number_of_frames > downsampling_factor, "Number of frames of the video should be larger than downsampling_factor"

        bgd = np.median(input_video[0:number_of_frames:downsampling_factor], axis=0)

        with imageio.get_writer(output_video_path, fps=fps) as writer:
            for frame in input_video:
                frame = np.abs(frame - bgd)
                if filter:
                    frame = utils.spatialFiltering(frame)
                writer.append_data(np.uint8(frame))


    def max_projection(self, input_video_path):
        """
        input_video_path : path to input video 

        return max projection frame of input video
        """
        cap = cv2.VideoCapture(input_video_path)
            
        if not cap.isOpened():
            print("Error: Cannot open video.")
            return
        
        max_proj_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if max_proj_frame is None:
                max_proj_frame = gray_frame
            else:
                max_proj_frame = np.maximum(max_proj_frame, gray_frame)
        
        cap.release()
        
        return np.uint8(max_proj_frame)


    def run_detections(self, input_video_path, detections_file_path, diameter = 15, minmass = 1000, batch_size = 100):
        """
        input_video_path : path to video for particles to be detected
        detections_file_path : detections save location, shoul have .csv extention
        diameter : see trackpy locate
        minmass : see trackpy locate 
        batch_size : function uses trackpy batch with batch_size to speed detection 

        takes input video file and detects particle locations using trackpy.locate (trackpy.batch
        for speed) and writes the output detections to a csv file with path detections_file
        """
        video = utils.gray(pims.Video(input_video_path))

        for start_frame in range(0, len(video), batch_size):
            end_frame = min(start_frame + batch_size, len(video)) 
            detections = trackpy.batch(frames=list(video[start_frame:end_frame]), diameter=diameter, minmass=minmass, engine='auto')
            mode = 'w' if start_frame == 0 else 'a'
            header = True if start_frame == 0 else False
            detections.to_csv(detections_file_path, mode=mode, header=header, index=False)
        
        if end_frame < len(video) - 1:
            detections = trackpy.batch(frames=list(video[end_frame:]), diameter=diameter, minmass=minmass, engine='auto')
            detections.to_csv(detections_file_path, mode='a', header=False, index=False)


    def compute_sampling_chamber_from_user_input(self, frame):
        """
        frame : frame with circular sampling chamber in it 

        takes user input (more than 3 points) to estimate the 
        center (x,y) and radius of the sampling chamber

        returns :
            sampling_chamber_info : sampling chamber dictionary with x, y, radius
        """

        click_points = []
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

        if(len(frame.shape)<3):
            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)

        # Function to handle mouse events
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                param.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow('Frame', frame)

        # Function to fit and draw a circle from points
        def fit_and_draw_circle(image, points):
            points_array = np.array(points, dtype=np.float32)
            x, y, radius, _ = taubinSVD(points_array)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image, center, radius, (0, 255, 0), 5)
            cv2.imshow('Fitted Circle', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return center[0], center[1], radius

        if frame is None:
            return None

        print("Please select sampling chamber boundary points for circle fitting") 
        print("Press SPACE bar to escape.")

        cv2.imshow('Frame', frame)
        cv2.setMouseCallback('Frame', click_event, click_points)

        while True:
            if cv2.waitKey(20) & 0xFF == 32:  # Spacebar ASCII
                break

        x, y, radius = fit_and_draw_circle(frame.copy(), click_points)
        sampling_chamber = {'x': x, 'y': y, 'radius': radius}

        return sampling_chamber


    def save_sampling_chamber_info(self, sampling_chamber_info, output_file_path):
        """
        saves sampling chamber info to output file path using pickle
        """
        with open(output_file_path, 'wb') as file:
            pickle.dump(sampling_chamber_info, file)


    def load_sampling_chamber_info(self, input_file_path):
        """
        loads sampling chamber info from file input_file_path

        returns : sampling chamber information in {'x': x, 'y': y, 'radius': radius} dictionary format
        """
        with open(input_file_path, 'rb') as file:
            sampling_chamber = pickle.load(file)

        return sampling_chamber


    def filter_outside_sampling_chamber(self, input_video_path, output_video_path, sampling_chamber_info, k=1.0, fps=30):
        """
        draws a circle to each frame in input video with centered at x,y and r = radius * k with sampling_chamber_info
        and filters outside of the sampling chamber, writes the outputs to output_video_path
        """
        fps = fps
        input_video = utils.gray(pims.Video(input_video_path))
        with imageio.get_writer(output_video_path, fps=fps) as writer:
            for frame in input_video:
                writer.append_data(np.uint8(utils.filterOutsideCircle(frame, circle=sampling_chamber_info, k=k)))

    
    def plot_detections_per_frame(self, detections):
        """
        detections : detections dataframe

        returns : 
            plots fig, ax = detections per frame and number of detections distribution plots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        detections_per_frame = detections.groupby('frame').size()
        detections_per_frame.plot(kind='bar', color='lightblue', width=1.0, ax=ax1)

        min_value = min(detections_per_frame)
        max_value = max(detections_per_frame)
        bin_edges = range(min_value, max_value + 2)

        ax1.set_title('Number of Detections per Frame')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Number of Detections')
        ax1.set_xticks([])
        ax1.set_ylim(0, 10 + int(np.median(detections_per_frame)))
        
        ax2.hist(detections_per_frame, bins=bin_edges, color='lightgreen', alpha=0.75)
        ax2.set_title('Distribution of Detections per Frame')
        ax2.set_xlabel('Number of Detections')
        ax2.set_ylabel('Number of Frames')

        fig.tight_layout()
        return fig


    def plot_detection_features(self, detections):
        """
        detections : detections dataframe

        returns : 
            plots figures of detection features visualized 
        """
        fig, axs= plt.subplots(5, 2, figsize=(8, 20))
        features = ['ecc', 'mass', 'size', 'signal', 'raw_mass']
        i_ax = 0
        for i, feature1 in enumerate(features):
            for feature2 in features[i+1:]:
                l, m = int(i_ax%5), int(i_ax/5)
                axs[l, m].scatter(detections[feature1].values, detections[feature2].values, color='lightblue', alpha=0.5)
                axs[l, m].set_title(f'{feature1} vs {feature2}')
                axs[l, m].set_xlabel(f'{feature1}')
                axs[l, m].set_ylabel(f'{feature2}')
                i_ax += 1

        fig.tight_layout()
        return fig


    def detections_projection(self, detections, width, height):
        """
        detections : detections dataframe
        width : frame width of original frame
        height : frame height of original frame

        returns : 
            frame with all detections plotted as a projection
        """
        canvas = np.zeros((height, width), dtype=np.uint64)
        plot_style = {'marker': 'o', 'markersize': 1}

        fig, ax = plt.subplots()
        ax.imshow(canvas, cmap='gray')
        trackpy.annotate(centroids=detections, image=canvas, color='b', plot_style=plot_style)

        fig.tight_layout()
        return fig


    def filter_detections(self, detections, filters):
        """
        detections : detections dataframe
        filters : dictionary of feature filtering values
        example : 
            filters = { 'ecc': {'threshold': 0.4, 'operation': 'lower'},     
            'mass': {'threshold': 3000, 'operation': 'higher'}
            } 

        returns : filtered_detections
        """
        filtered_detections = detections.copy()
        # for each filtering operations in filters
        for col, filter in filters.items():
            if col in detections.columns:
                threshold = filter['threshold']
                operation = filter['operation']

                if operation == 'lower':
                    filtered_detections = filtered_detections[detections[col] < threshold]
                elif operation == 'higher':
                    filtered_detections = filtered_detections[detections[col] > threshold]

                filtered_detections.reset_index(drop=True, inplace=True)

        return filtered_detections

    
    def track_cells(self, 
                   detections, 
                   cell_config_file, 
                   tracking_features = ['ecc'], 
                   max_displacement=50, 
                   ):
        """
        Loads cell_config_file generates tracking objects and runs tracking on 
        detections. 

        tracking_features = includes these features in tracking 
        """
        tracking_objects = self._generate_tracking_objects(detections=detections)
        tracks = self._run_tracking(tracking_objects=tracking_objects,
                                    tracking_features=tracking_features,
                                    cell_config_file=cell_config_file,
                                    max_displacement=max_displacement)
        tracks_list = []
        for tr in tracks:
            tracks_list.append(pd.DataFrame(tr.to_dict()))
        
        return pd.concat(tracks_list, ignore_index=True)


    def _generate_tracking_objects(self, detections):
        """
        Converts trackpy detections dataframe to btrack object list

        returns btrack objects list
        """
        df = pd.DataFrame({
            't': detections['frame'].values,
            'x': detections['x'].values,
            'y': detections['y'].values,
            'z': np.zeros_like(detections['x'].values),
            'mass': detections['mass'].values,
            'ecc': detections['ecc'].values,
            'size': detections['size'].values,
            'raw_mass': detections['raw_mass'].values,
            'signal': detections['signal'].values
        })

        # Process the DataFrame to create tracking objects
        tracking_objects = []
        for idx, row in df.iterrows():
            data = row.to_dict()
            data['ID'] = idx
            
            obj = btypes.PyTrackObject.from_dict(data)
            obj._properties = {key: data[key] for key in ['mass', 'ecc', 'size', 'raw_mass', 'signal']}
            obj.properties = obj._properties
            
            tracking_objects.append(obj)

        return tracking_objects


    def _run_tracking(self, tracking_objects, tracking_features, cell_config_file, max_displacement=50):
        """
        implements tracking on tracking_objects using btrack tracker
        tracking_objects : list of btrack objects
        cell_config_file : configurations file for btrack, 
        look at btrack modules doc for more info

        returns : tracks dataframe
        """
        with btrack.BayesianTracker() as tracker:
            # configure the tracker using a config file
            tracker.configure(cell_config_file)
            tracker.max_search_radius = max_displacement
            tracker.tracking_updates = ["MOTION"]
            tracker.features = tracking_features

            # append the objects to be tracked
            tracker.append(tracking_objects)

            # set the tracking volume
            tracker.volume=((0, 1504), (0, 1088))

            # track them (in interactive mode)
            tracker.track(step_size=100)

            # generate hypotheses and run the global optimizer
            tracker.optimize()

            return tracker.tracks
        

    def _filter_tracks_by_time_and_area(self, tracks, min_track_length, min_convex_hull_covered):
        """
        Filters detections by min_track_length (e.g. min_track_length=1800 means 
        the minimum length of 1 minute for a 30 fps video) and min_convex_hull_covered 
        by the track

        input : 
            tracks : all tracks found by btrack tracker in dataframe

        returns : 
            selected_indices : selected track indices after filtering
        """
        selected_ids = []
        object_ids = np.unique(tracks['ID'].values)
        for id in object_ids:
            track = tracks[tracks['ID'] == id]
            x = track['x'].values
            y = track['y'].values
            t = track['t'].values
            xy = np.stack((x, y), axis=1)
            delta_t = np.max(t) - np.min(t)
            if delta_t > min_track_length:
                try:
                    hull = ConvexHull(xy)
                    if hull.area > min_convex_hull_covered:
                        selected_ids.append(id)
                except Exception as e:
                    pass
        return selected_ids


    def filter_tracks(self, tracks, min_track_length=1800, min_convex_hull_covered=500):
        """
        Filters detections by min_track_length (e.g. min_track_length=1800 means 
        the minimum length of 1 minute for a 30 fps video) and min_convex_hull_covered 
        by the track

        input : 
            tracks : all tracks found by btrack tracker in dataframe

        returns : 
            filtered tracks dataframe
        """
        selected_ids = self._filter_tracks_by_time_and_area(tracks, min_track_length, min_convex_hull_covered)
        return tracks[tracks['ID'].isin(selected_ids)] # .reset_index(drop=True, inplace=True)

        
    def connect_tracks(self, tracks, 
                       consider_for_connecting_track_length_threshold=30, 
                       consider_for_connecting_convex_hull_threshold=50, 
                       max_displacement=200, memory=200):
        """
        Connects tracks

        inputs : 
            tracks : tracks dataframe
            consider_for_connecting_track_length_threshold : select track for connecting if track length is larger than this threshold
            consider_for_connecting_convex_hull_threshold : select track for connecting if track convex hull is larger than this threshold
            max_displacement : max displacement between track ending and starting for connecting
            memory : max time between track ending and starting for connecting

        returns : 
            connected tracks dataframe
        """
        selected_ids = self._filter_tracks_by_time_and_area(tracks=tracks, 
                                                            min_track_length=consider_for_connecting_track_length_threshold, 
                                                            min_convex_hull_covered=consider_for_connecting_convex_hull_threshold)
        next_id = 0
        equal_tracks = {}

        if selected_ids:
            equal_tracks[f'{next_id}'] = [selected_ids[0]]

            for i in range(1, len(selected_ids)):
                prev_id = selected_ids[i - 1]
                curr_id = selected_ids[i]
                previous_track = tracks.loc[tracks['ID'] == prev_id]
                current_track = tracks.loc[tracks['ID'] == curr_id]
                distance = np.linalg.norm(previous_track[['x', 'y']].values[-1] - current_track[['x', 'y']].values[0])

                if distance < max_displacement:
                    equal_tracks[f'{next_id}'].append(curr_id)
                else:
                    next_id += 1
                    equal_tracks[f'{next_id}'] = [curr_id]

        for new_id, ids in equal_tracks.items():
            tracks.loc[tracks['ID'].isin(ids), 'ID'] = int(new_id)

        return tracks


    def plot_track_on_frame(self, frame, track):
        """
        returns track plotted on frame
        """
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(frame, extent=[0, frame.shape[1], frame.shape[0], 0])
        ax.scatter(track['x'].values, track['y'].values, c='red', s=1)
        
        return fig


    def plot_track(self, track, sampling_chamber_info = None):
        """
        returns track plot
        draws sampling chamber on frame if sampling_chamber_info is not None
        """
        fig, ax = plt.subplots(figsize=(24, 18))
        if sampling_chamber_info:
            ax.add_patch(patches.Circle((sampling_chamber_info['x'], 
                                         sampling_chamber_info['y']),
                                        int(sampling_chamber_info['radius']),
                                        color='gray',
                                        linewidth=5,
                                        fill=False))
            
        x = track['x'].values
        y = track['y'].values
        scatter = ax.scatter(x, y, c=track.t, cmap='inferno')

        cbar = fig.colorbar(scatter, ax=ax) 
        cbar.set_label('t(s)', fontsize=28, weight='bold')

        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.invert_yaxis()

        return fig
