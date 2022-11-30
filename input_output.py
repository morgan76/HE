# -*- coding: utf-8 -*-
"""
A lot of functions are copied from MSAF:
https://github.com/urinieto/msaf/blob/master/msaf/input_output.py
These set of functions help the algorithms of MSAF to read and write
files of the Segmentation Dataset.
"""
import datetime
import glob
import json
import logging
import os
import re
from pathlib import Path
from collections import defaultdict

import ujson
import numpy as np
import jams
import six
import mir_eval

# Local stuff
import utils
from configuration import config

# Put dataset config in a global var
ds_config = config.dataset


class FileStruct:
    def __init__(self, audio_file):
        audio_file = Path(audio_file)
        self.track_name = audio_file.stem
        self.audio_file = audio_file
        self.ds_path = audio_file.parents[1]
        self.json_file = self.ds_path.joinpath('features', self.track_name
                                               + '.json')
        self.ref_file = self.ds_path.joinpath('references', self.track_name
                                              + ds_config.references_ext)
        self.est_file = self.ds_path.joinpath('estimations', self.track_name 
                                              + ds_config.estimations_ext)
        self.beat_file = self.ds_path.joinpath('features', self.track_name+'_beats_'
                                              + '.json')
        self.estimates_file = self.ds_path.joinpath('references/estimates/', self.track_name
                                              + ds_config.references_ext)
        self.chroma_file = self.ds_path.joinpath('features/chroma/', self.track_name
                                              + '.npy')
        

    def __repr__(self):
        """Prints the file structure."""
        return "FileStruct(\n\tds_path=%s,\n\taudio_file=%s,\n\test_file=%s," \
            "\n\json_file=%s,\n\tref_file=%s\n)" % (
                self.ds_path, self.audio_file, self.est_file,
                self.json_file, self.ref_file)

    def get_feat_filename(self, feat_id):
        return self.ds_path.joinpath('features', feat_id,
                                     self.track_name + '.npy')


def read_estimations(est_file, boundaries_id, labels_id=None, **params):
    """Reads the estimations (boundaries and/or labels) from a jams file
    containing the estimations of an algorithm.

    Parameters
    ----------
    est_file : str
        Path to the estimated file (JAMS file).
    boundaries_id : str
        Identifier of the algorithm used to compute the boundaries.
    labels_id : str
        Identifier of the algorithm used to compute the labels.
    params : dict
        Additional search parameters. E.g. {"feature" : "pcp"}.

    Returns
    -------
    boundaries : np.array((N,2))
        Array containing the estimated boundaries in intervals.
    labels : np.array(N)
        Array containing the estimated labels.
        Empty array if labels_id is None.
    """
    # Open file and read jams
    jam = jams.load(est_file)

    # Find correct estimation
    est = find_estimation(jam, boundaries_id, labels_id, params)
    if est is None:
        raise IOError("No estimations for file: %s" % est_file)

    # Get data values
    all_boundaries, all_labels = est.to_interval_values()

    if params["hier"]:
        hier_bounds = defaultdict(list)
        hier_labels = defaultdict(list)
        for bounds, labels in zip(all_boundaries, all_labels):
            level = labels["level"]
            hier_bounds[level].append(bounds)
            hier_labels[level].append(labels["label"])
        # Order
        all_boundaries = []
        all_labels = []
        for key in sorted(list(hier_bounds.keys())):
            all_boundaries.append(np.asarray(hier_bounds[key]))
            all_labels.append(np.asarray(hier_labels[key]))

    return all_boundaries, all_labels


def read_references_estimates(audio_path, algo_id=0):

    ds_path = os.path.dirname(os.path.dirname(audio_path))
    try:
        jam_path = os.path.join(ds_path, ds_config.references_dir+'/estimates/',
                                os.path.basename(audio_path) +
                                ds_config.references_ext)
        jam = jams.load(jam_path, validate=False)
    except:
        try:
            jam_path = os.path.join(ds_path, ds_config.references_dir+'/estimates/',
                                    os.path.basename(audio_path)[:-4] +
                                    ds_config.references_ext)
            jam = jams.load(jam_path, validate=False)
        except:
            jam_path = os.path.join(ds_path, ds_config.references_dir+'/estimates/',
                                    os.path.basename(audio_path)[:-5] +
                                    ds_config.references_ext)
            jam = jams.load(jam_path, validate=False)
    ann = jam.search(namespace='segment_.*')[algo_id]
    ref_inters, ref_labels = ann.to_interval_values()
    # Intervals to times
    ref_times = utils.intervals_to_times(ref_inters)
       
    return ref_times, ref_labels
    

def get_ref_labels(file_struct, level, annot, config):
    if config.ds_path == None:
        config.ds_path = str(file_struct.ds_path)
    if 'SALAMI_2annot' in config.ds_path or 'SALAMI_left' in config.ds_path or 'SALAMI_test_MIREX' in config.ds_path:
        ref_inters_list, ref_labels_list, duration = read_references_2annot(file_struct.audio_file, annot)
        ref_times = utils.intervals_to_times(ref_inters_list[level])
        ref_labels = ref_labels_list[level]
    else:
        jam = jams.load(str(file_struct.ref_file), validate=False)
        duration = jam.file_metadata.duration
        ref_times, ref_labels = read_references(file_struct.audio_file, False)
    ref_times, ref_labels = utils.remove_empty_segments(ref_times, ref_labels)
    return ref_labels, ref_times, duration



def read_references(audio_path, annotator_id=0, hier=False):
    """Reads the boundary times and the labels.

    Parameters
    ----------
    audio_path : str
        Path to the audio file

    Returns
    -------
    ref_times : list
        List of boundary times
    ref_labels : list
        List of labels

    Raises
    ------
    IOError: if `audio_path` doesn't exist.
    """
    # Dataset path
    ds_path = os.path.dirname(os.path.dirname(audio_path))


    try:
        jam_path = os.path.join(ds_path, ds_config.references_dir,
                                os.path.basename(audio_path)[:-4] +
                                ds_config.references_ext)
        

        jam = jams.load(jam_path, validate=False)
    except:
        jam_path = os.path.join(ds_path, ds_config.references_dir,
                                os.path.basename(audio_path)[:-5] +
                                ds_config.references_ext)
            

        jam = jams.load(jam_path, validate=False)



    ##################
    low = True # Low parameter for SALAMI
    ##################


    if not hier:
        if low:  
            try:
                ann = jam.search(namespace='segment_salami_lower.*')[0]
            except:
                try:
                    ann = jam.search(namespace='segment_salami_upper.*')[0]
                except:
                    ann = jam.search(namespace='segment_.*')[annotator_id]
        else:
            try:
                ann = jam.search(namespace='segment_salami_upper.*')[0]
            except:
                ann = jam.search(namespace='segment_.*')[annotator_id]
        
        ref_inters, ref_labels = ann.to_interval_values()
        # Intervals to times
        ref_times = utils.intervals_to_times(ref_inters)
        
        #ref_times = [*[0.000] , *ref_times]
        
        #ref_labels = [*['Silence'], *ref_labels]
        
        return ref_times, ref_labels

    else:

        list_ref_times, list_ref_labels = [], []
        upper = jam.search(namespace='segment_salami_upper.*')[0]
        ref_inters_upper, ref_labels_upper = upper.to_interval_values()
        
        list_ref_times.append(utils.intervals_to_times(ref_inters_upper))
        list_ref_labels.append(ref_labels_upper)

        annotator = upper['annotation_metadata']['annotator']
        lowers = jam.search(namespace='segment_salami_lower.*')

        for lower in lowers:
            if lower['annotation_metadata']['annotator'] == annotator:
                ref_inters_lower, ref_labels_lower = lower.to_interval_values()
                list_ref_times.append(utils.intervals_to_times(ref_inters_lower))
                list_ref_labels.append(ref_labels_lower)

        return list_ref_times, list_ref_labels





def read_references_2annot(audio_path, index):
    ds_path = os.path.dirname(os.path.dirname(audio_path))

    # Read references
    try:
        jam_path = os.path.join(ds_path, ds_config.references_dir,
                                os.path.basename(audio_path)[:-4] +
                                ds_config.references_ext)
        

        jam = jams.load(jam_path, validate=False)
    except:
        jam_path = os.path.join(ds_path, ds_config.references_dir,
                                os.path.basename(audio_path)[:-5] +
                                ds_config.references_ext)
        

        jam = jams.load(jam_path, validate=False)


    list_ref_times, list_ref_labels = [], []
    upper = jam.search(namespace='segment_salami_upper.*')[index]
    ref_inters_upper, ref_labels_upper = upper.to_interval_values()
    duration = jam.file_metadata.duration
    ref_inters_upper = utils.intervals_to_times(ref_inters_upper)
    #
    ref_inters_upper, ref_labels_upper = utils.remove_empty_segments(ref_inters_upper, ref_labels_upper)
    ref_inters_upper = utils.times_to_intervals(ref_inters_upper)
    (ref_inters_upper, ref_labels_upper) = mir_eval.util.adjust_intervals(ref_inters_upper, ref_labels_upper, t_min=0, t_max=duration)
    list_ref_times.append(ref_inters_upper)
    list_ref_labels.append(ref_labels_upper)


    lower = jam.search(namespace='segment_salami_lower.*')[index]
    ref_inters_lower, ref_labels_lower = lower.to_interval_values()
    ref_inters_lower = utils.intervals_to_times(ref_inters_lower)
    #(ref_inters_lower, ref_labels_lower) = mir_eval.util.adjust_intervals(ref_inters_lower, ref_labels_lower, t_min=0, t_max=duration)
    ref_inters_lower, ref_labels_lower = utils.remove_empty_segments(ref_inters_lower, ref_labels_lower)
    ref_inters_lower = utils.times_to_intervals(ref_inters_lower)
    (ref_inters_lower, ref_labels_lower) = mir_eval.util.adjust_intervals(ref_inters_lower, ref_labels_lower, t_min=0, t_max=duration)
    list_ref_times.append(ref_inters_lower)
    list_ref_labels.append(ref_labels_lower)



    return list_ref_times, list_ref_labels, duration




def align_times(times, frames):
    """Aligns the times to the closest frame times (e.g. beats).

    Parameters
    ----------
    times: np.ndarray
        Times in seconds to be aligned.
    frames: np.ndarray
        Frame times in seconds.

    Returns
    -------
    aligned_times: np.ndarray
        Aligned times.
    """
    dist = np.minimum.outer(times, frames)
    bound_frames = np.argmax(np.maximum(0, dist), axis=1)
    aligned_times = np.unique(bound_frames)
    return aligned_times



def get_dataset_files(in_path):
    """Gets the files of the given dataset."""
    # Get audio files
    audio_files = []
    for ext in ds_config.audio_exts:
        audio_files += glob.glob(
            os.path.join(in_path, ds_config.audio_dir, "*" + ext))

    # Make sure directories exist
    utils.ensure_dir(os.path.join(in_path, ds_config.features_dir))
    utils.ensure_dir(os.path.join(in_path, ds_config.estimations_dir))
    utils.ensure_dir(os.path.join(in_path, ds_config.references_dir))

    # Get the file structs
    file_structs = []
    for audio_file in audio_files:
        file_structs.append(FileStruct(audio_file))

    # Sort by audio file name
    file_structs = sorted(file_structs,
                          key=lambda file_struct: file_struct.audio_file)

    return file_structs


def read_hier_references(audio_path, annotation_id=0, exclude_levels=[]):
    """Reads hierarchical references from a jams file.

    Parameters
    ----------
    jams_file : str
        Path to the jams file.
    annotation_id : int > 0
        Identifier of the annotator to read from.
    exclude_levels: list
        List of levels to exclude. Empty list to include all levels.

    Returns
    -------
    hier_bounds : list
        List of the segment boundary times in seconds for each level.
    hier_labels : list
        List of the segment labels for each level.
    hier_levels : list
        List of strings for the level identifiers.
    """

    ds_path = os.path.dirname(os.path.dirname(audio_path))

    hier_bounds = []
    hier_labels = []
    
    namespaces = ["segment_salami_upper", "segment_salami_function",
                  "segment_open", "segment_tut", "segment_salami_lower", "multi_segment"]

    ds_path = os.path.dirname(os.path.dirname(audio_path))

    low = True
    if 'SALAMI' in ds_path:
        if low: 
            try:
                jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier_low',
                                        os.path.basename(audio_path)[:-4] +
                                        ds_config.references_ext)
                
                jam = jams.load(jam_path, validate=False)
            except:
                jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier_low',
                                        os.path.basename(audio_path)[:-5] +
                                        ds_config.references_ext)
        else:
            try:
                jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier_up',
                                        os.path.basename(audio_path)[:-4] +
                                        ds_config.references_ext)
                
                jam = jams.load(jam_path, validate=False)
            except:
                jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier_up',
                                        os.path.basename(audio_path)[:-5] +
                                        ds_config.references_ext)
    else:

        # Read references
        try:
            jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier',
                                    os.path.basename(audio_path)[:-4] +
                                    ds_config.references_ext)
            
            jam = jams.load(jam_path, validate=False)
        except:
            jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier',
                                    os.path.basename(audio_path)[:-5] +
                                    ds_config.references_ext)

    jam = jams.load(jam_path, validate=False)
    
    duration = jam.file_metadata.duration
    # Remove levels if needed
    for exclude in exclude_levels:
        if exclude in namespaces:
            namespaces.remove(exclude)

    # Build hierarchy references
    for i in jam['annotations']['multi_segment']:
        bounds_0, labels_0 = [], []
        bounds_1, labels_1 = [], []
        bounds_2, labels_2 = [], []
        for bound in i['data']:
            if bound.value['level'] == 0:
                bounds_0.append(bound.time)
                labels_0.append(bound.value['label'])
            elif bound.value['level'] == 1:
                bounds_1.append(bound.time)
                labels_1.append(bound.value['label'])
            elif bound.value['level'] == 2:
                bounds_2.append(bound.time)
                labels_2.append(bound.value['label'])
        if len(bounds_0) > 0:
            hier_bounds.append(bounds_0)
            hier_labels.append(labels_0)
        if len(bounds_1) > 0:
            hier_bounds.append(bounds_1)
            hier_labels.append(labels_1)
        if len(bounds_2) > 0:
            hier_bounds.append(bounds_2)
            hier_labels.append(labels_2)
    
    #print('Input output =', [len(i) for i in hier_bounds], [i for i in hier_labels])
    ref_inters_list = []
    for ref_int, ref_lab in zip(hier_bounds, hier_labels):
        #
        ref_int = utils.times_to_intervals(ref_int)
        (ref_int, ref_lab) = mir_eval.util.adjust_intervals(ref_int, ref_lab, t_min=0, t_max=duration)
        #ref_int, ref_lab = utils.remove_empty_segments(ref_int, ref_lab)
        ref_inters_list.append(ref_int)
    
    
    return ref_inters_list, hier_labels


def get_duration(features_file):
    """Reads the duration of a given features file.

    Parameters
    ----------
    features_file: str
        Path to the JSON file containing the features.

    Returns
    -------
    dur: float
        Duration of the analyzed file.
    """
    with open(features_file) as f:
        feats = json.load(f)
    return float(feats["globals"]["dur"])


def write_mirex(times, labels, out_file):
    """Writes results to file using the standard MIREX format.

    Parameters
    ----------
    times: np.array
        Times in seconds of the boundaries.
    labels: np.array
        Labels associated to the segments defined by the boundaries.
    out_file: str
        Output file path to save the results.
    """
    inters = utils.times_to_intervals(times)
    assert len(inters) == len(labels)
    out_str = ""
    for inter, label in zip(inters, labels):
        out_str += "%.3f\t%.3f\t%s\n" % (inter[0], inter[1], label)
    with open(out_file, "w") as f:
        f.write(out_str[:-1])


def write_beats_(beat_times, file_struct):

    # Construct a new JAMS object and annotation records
    
    try:
        jam = jams.load(file_struct.ref_file)
    except:
        jam = jams.JAMS()
    # Store the track duration
    #jam.file_metadata.duration = 0

    beat_a = jams.Annotation(namespace='beat')
    beat_a.annotation_metadata = jams.AnnotationMetadata()
    # Add beat timings to the annotation record.
    # The beat namespace does not require value or confidence fields,
    # so we can leave those blank.
    for t in beat_times:
        beat_a.append(time=t, duration=0.0)

    # Store the new annotation in the jam
    jam.annotations.append(beat_a)
    # Save to disk
    jam.save(str(file_struct.ref_file))


def write_beats(beat_times, file_struct, feat_config):

    # Construct a new JAMS object and annotation records
    
    # Save feature configuration in JSON file
    json_file = file_struct.json_file
    if json_file.exists() and os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
    else:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        out_json = utils.create_json_metadata(file_struct.audio_file, 0, feat_config)
    out_json["est_beats"] = list(beat_times)
    with open(json_file, "w") as f:
        ujson.dump(out_json, f, indent=4)


def write_downbeats(downbeat_times, jam_file):

    try:
        beat_times = read_beats(jam_file)
    except IOError:
        raise IOError('To save downbeats, you should have computed and saved '
                      'beats beforehand.')
    # Construct a new JAMS object and annotation records
    jam = jams.JAMS()

    # Store the track duration
    jam.file_metadata.duration = 0

    beatpos_a = jams.Annotation(namespace='beat_position')
    beatpos_a.annotation_metadata = jams.AnnotationMetadata()

    # Add downbeat timings to the annotation record.
    beatpos_frame = utils.align_beats_downbeats(beat_times, downbeat_times)

    for row in beatpos_frame.itertuples():
        # row.num_beats = 2
        value= {'beat_units': row.beat_units,
                'measure': row.measure,
                'num_beats': row.num_beats,
                'position': row.position}
        beatpos_a.append(time=row.time,
                         duration=row.duration,
                         value=value)

    # Store the new annotation in the jam
    jam.annotations.append(beatpos_a)

    # Save to disk
    jam.save(str(jam_file))


def read_beats(json_file):
    with open(json_file, 'r') as f:
        out_json = ujson.load(f)
    beat_strings = out_json["est_beats"].split('[')[1].split(']')[0].split(',')
    if len(beat_strings) < 10:
        return []
    else:
        beat_times = [int(i) for i in beat_strings]
    return beat_times

def read_beats_(jam_file):
    jam = jams.load(str(jam_file))
    annot = jam.search(namespace='beat')[0]
    beat_times = annot.to_event_values()[0]
    return beat_times


def read_beats_downbeats(jam_file):
    jam = jams.load(str(jam_file))
    annot = jam.search(namespace='beat_position')[0]

    events = annot.to_event_values()
    beat_times = events[0]
    downbeat_times = [events[0][i] for i in range(len(beat_times)) if events[1][i]['position'] == 1]
    return beat_times, downbeat_times


def write_features(features, file_struct, feat_id, feat_config, beat_frames, duration=None):
    # Save actual feature file in .npy format
    feat_file = file_struct.get_feat_filename(feat_id)
    json_file = file_struct.json_file
    feat_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_file, features)
    print('File saved')

    # Save feature configuration in JSON file
    if json_file.exists() and os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
    else:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        out_json = utils.create_json_metadata(file_struct.audio_file, duration,
                                        feat_config)
    out_json[feat_id] = {}
    variables = vars(getattr(feat_config, feat_id))
    for var in variables:
        out_json[feat_id][var] = str(variables[var])
    with open(json_file, "w") as f:
        ujson.dump(out_json, f, indent=4)

    
    json_file = file_struct.beat_file
    if beat_frames != []:
        if json_file.exists():
            with open(json_file, 'r') as f:
                out_json = ujson.load(f)
        else:
            json_file.parent.mkdir(parents=True, exist_ok=True)
            out_json = utils.create_json_metadata(file_struct.audio_file, duration,
                                            feat_config)
        out_json['est_beats'] = json.dumps([int(i) for i in beat_frames])
        with open(json_file, "w") as f:
            ujson.dump(out_json, f, indent=4)


def update_beats(file_struct, feat_config, beat_frames, duration):
    json_file = file_struct.beat_file
    if json_file.exists():
        print('BEAT FILE EXISTS')
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
            print(out_json['est_beats'])
    else:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        out_json = utils.create_json_metadata(file_struct.audio_file, duration,
                                        feat_config)
    out_json['est_beats'] = json.dumps([int(i) for i in beat_frames])
    with open(json_file, "w") as f:
        ujson.dump(out_json, f, indent=4)

    if json_file.exists():
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
            print(out_json['est_beats'])
    


