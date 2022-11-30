import numpy as np
import jams
import mir_eval
import librosa
import os

from input_output import FileStruct
from utils import times_to_intervals, intervals_to_times, remove_empty_segments
import input_output as io
from algorithms.scluster.main2 import do_segmentation as scluster
from embed import eval_track as embed





def eval_segmentation(audio_file, embedding_net, config, device, feat_id, return_data=False):
    # REFERENCE
    feat_id = config.feat_id
    ref_file = FileStruct(audio_file).ref_file

    if os.path.isfile(str(ref_file)):
        # loading annotations
        file_struct = FileStruct(audio_file)
        ref_labels, ref_times, duration = io.get_ref_labels(file_struct, config.annot_level, config.annotator_id, config)
        ref_inter = times_to_intervals(ref_times)
        (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=duration)
        
        # loading beat times
        beat_frames = io.read_beats(FileStruct(audio_file).beat_file)
        beat_times = librosa.frames_to_time(beat_frames, sr=config.sample_rate, hop_length=config.hop_length)
        
        # calculating embeddings
        embeddings = embed(audio_file, embedding_net, config, device, feat_id)
        
            
        temp_F3, temp_R3, temp_P3, temp_PFC, temp_S_F = [], [], [], [], []
        temp_F1, temp_R1, temp_P1 = [], [], []
        est_inter_list, est_labels_list, Cnorm = scluster(embeddings.T, embeddings.T, True)

        for est_idxs, est_labels in zip(est_inter_list, est_labels_list):
            est_idxs = [beat_times[int(i)] for i in est_idxs]
            # Converting and cleaning boundary predictions to intervals 
            est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
            est_inter = times_to_intervals(est_idxs)
            est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0, t_max=duration)
            
            P1, R1, F1 = mir_eval.segment.detection(ref_inter,est_inter,window=.5,trim=True)                                              

            P3, R3, F3 = mir_eval.segment.detection(ref_inter,est_inter,window=3,trim=True) 

            precision, recall, f_PFC = mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels)
            S_over, S_under, S_F = mir_eval.segment.nce(ref_inter, ref_labels, est_inter, est_labels)
            
            temp_S_F.append(S_F)
            temp_PFC.append(f_PFC)
            temp_P3.append(P3)
            temp_R3.append(R3)
            temp_F3.append(F3)
            temp_P1.append(P1)
            temp_R1.append(R1)
            temp_F1.append(F1)
            ssm, novelty_curve = 0, 0
        
        max_ind = np.argmax(temp_F3)
        P3 = temp_P3[max_ind]
        R3 = temp_R3[max_ind]
        F3 = temp_F3[max_ind]
        P1 = temp_P1[max_ind]
        R1 = temp_R1[max_ind]
        F1 = temp_F1[max_ind]
        PFC = temp_PFC[max_ind]
        S_F = temp_S_F[max_ind]
        est_inter = est_inter_list[max_ind]
        
        est_labels = est_labels_list[max_ind]
        est_inter = times_to_intervals([beat_times[int(i)] for i in est_inter])
                
            
        results1 = {"P1": P1,"R1": R1,"F1": F1}
        results2 = {"PFC": PFC,"S_F": S_F}
        results3 = {"P3": P3,"R3": R3,"F3": F3}


        if return_data:
            return results1, results2, results3, ssm, novelty_curve, embeddings, ref_inter, est_inter, ref_labels, est_labels
        else:
            return results1, results2, results3
