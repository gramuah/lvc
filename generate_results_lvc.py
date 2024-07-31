#!/bin/python
# Live Video Captioning Evaluation script
# ----------------------------------------
# evaluation scripts for live video captioning, support python 3
# Modified from https://github.com/ranjaykrishna/densevid_eval/blob/master/evaluate.py
# --------------------------------------------------------
# Live Video Captioning Evaluation script
# This script includes the Live Score (LS) metrics described in the paper
# Licensed under GPL-3.0 license [see LICENSE for details]
# Written by Eduardo Blanco-Fernández and Roberto J. López-Sastre
# --------------------------------------------------------


import io
import sys
from tqdm import tqdm
import json
import os
import random
import string
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

Set = set
import numpy as np


def random_string(string_length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(string_length))

def remove_nonascii(text):
    return "".join([i if ord(i) < 128 else " " for i in text])
        
        
class ANETcaptions(object):
    """
        Initializes the ANETcaptions object.

        Args:
            ground_truth_filenames (list): List of ground truth filenames.
            prediction_filename (str): Filename of the prediction file.
            tious (list): List of tIoU values to be used for evaluation.
            max_proposals (int): Maximum number of proposals per video.
            prediction_fields (list): List of prediction fields.
            verbose (bool): Whether to print verbose output.
            frames (None): Not used.
            delta_t_window (int): Temporal window for the analysis of the LVC model in history mode.
        """
    PREDICTION_FIELDS = ["results", "version", "external_data"]

    def __init__(
        self,
        ground_truth_filenames=None,
        prediction_filename=None,
        tious=[0.001], #for LVC only one minimal tIoU is used
        max_proposals=1000,
        prediction_fields=PREDICTION_FIELDS,
        verbose=False,
        frames=None,
        delta_t_window=5, #LVC _ temporal window for the analysis of the LVC model in history mode
    ):
        # Check that the gt and submission files exist and load them
        if len(tious) == 0:
            raise IOError("Please input a valid tIoU.")
        if not ground_truth_filenames:
            raise IOError("Please input a valid ground truth file.")
        if not prediction_filename:
            raise IOError("Please input a valid prediction file.")
        cwd = os.path.abspath(os.path.dirname(__file__))
        self.tokenizer = PTBTokenizer()
        self.verbose = verbose
        self.tious = [0.001]
        self.max_proposals = 1000
        self.delta_t_window = delta_t_window
        self.frames = frames
        self.max_proposals = max_proposals
        self.pred_fields = prediction_fields
        self.weighted = False  # True if the weighted version of the Live Scorer (wLS) is used
        self.history = False  # True if the version with history of the Live Scorer (hLS) is used
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        self.ground_truths_keys = [vid for gt in self.ground_truths for vid in gt]
        print(
            "available video number",
            len(set(self.ground_truths_keys) & set(self.prediction.keys())),
        )

        self.scorers = [
            (Bleu(1), ["Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
        ]

    def import_prediction(self, prediction_filename):
        """
        Imports a prediction file and returns a dictionary of results.

        Args:
            prediction_filename (str): The path to the prediction file.

        Returns:
            dict: A dictionary containing the results, where each video ID is mapped to a list of proposals.
        """
        if self.verbose:
            print("| Loading submission...")
        submission = json.load(open(prediction_filename))
        if not all([field in submission.keys() for field in self.pred_fields]):
            raise IOError("Please input a valid ground truth file.")
        # Ensure that every video is limited to the correct maximum number of proposals.
        results = {}
        for vid_id in submission["results"]:
            results[vid_id] = submission["results"][vid_id][: self.max_proposals]
        return results

    def import_ground_truths(self, filenames):
        """
        Imports ground truths from the given filenames.

        Args:
            filenames (list): A list of filenames containing the ground truth data.

        Returns:
            list: A list of dictionaries, where each dictionary represents the ground truth data from a file.

        """
        cwd = os.path.abspath(os.path.dirname(__file__))
        gts = []
        self.n_ref_vids = Set()
        for filename in filenames:
            gt = json.load(open(filename))
            self.n_ref_vids.update(gt.keys())
            gts.append(gt)
        if self.verbose:
            print(
                "| Loading GT. #files: %d, #videos: %d"
                % (len(filenames), len(self.n_ref_vids))
            )
        return gts

    def iou(self, interval_1, interval_2):
        """
        Calculates the Intersection over Union (IoU) between two intervals.

        Args:
            interval_1 (tuple): The first interval represented as a tuple (start, end).
            interval_2 (tuple): The second interval represented as a tuple (start, end).

        Returns:
            float: The IoU value between the two intervals.

        """
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(
            max(end, end_i) - min(start, start_i), end - start + end_i - start_i
        )
        iou = float(intersection) / (union + 1e-8)
        return iou

    def check_gt_exists(self, vid_id):
        """
        Check if a given video ID exists in the ground truths.

        Parameters:
        - vid_id (str): The video ID to check.

        Returns:
        - bool: True if the video ID exists in the ground truths, False otherwise.
        """
        for gt in self.ground_truths:
            if vid_id in gt:
                return True
        return False

    def get_gt_vid_ids(self):
        """
        Get the list of video IDs from the ground truths.

        Returns:
            list: A list of video IDs.
        """
        vid_ids = set([])
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)

    def evaluate_video(self, vid_id, gts, weighted, history): #Live Scorer (LS) Code
        delta_t_mem = self.delta_t_window
        pred = self.prediction[vid_id]
        short_videos = [
            (video_pred["sentence"], video_pred["timestamp"])
            for i, video_pred in enumerate(pred)
        ]
        # Compute number of segments of duration of delta t for the GT
        num_delta_t = len(
            pred
        )
        live_scores = []
        fp_count = 0  # Count false positives

        for delta_t in range(num_delta_t):
            score_tiou_list, fp = self.evaluate_tiou(short_videos[delta_t], vid_id, gts)
            # print(score_tiou_list)
            if history:
                delta_window = delta_t + 1 if delta_t < delta_t_mem else delta_t_mem
                window_scores = np.array(
                    live_scores[-(delta_t_mem - 1) :]
                    if live_scores
                    else [0] * min(5, len(live_scores))
                )

                score_tiou_mean = (
                    sum(window_scores) + np.array(score_tiou_list)
                ) / delta_window
            else:
                score_tiou_mean = (
                    (np.sum(live_scores, axis=0) + np.array(score_tiou_list))
                    / (delta_t + 1)
                    if delta_t != 0
                    else np.array(score_tiou_list)
                )  # Calculamos la media, sin pesado
            if weighted:
                fp_count += (
                    1 if not fp else 0
                )  # La variable fp será "false" si se detecta un false positive
                if history:
                    # Modificamos el máximo de false positives contados que puede haber,
                    # ya que la máxima ventana es 5
                    if fp_count > (delta_t_mem - 1):
                        fp_count = (delta_t_mem - 1) if fp else delta_t_mem
                    fpw = fp_count / (
                        (delta_t + 1) if delta_t <= (delta_t_mem - 1) else delta_t_mem
                    )
                else:
                    fpw = fp_count / ((delta_t + 1) if delta_t != 0 else 1)
                score_tiou_mean *= np.exp(-fpw)
            live_scores.append(score_tiou_mean.tolist())

        return live_scores

    def evaluate_tiou(self, pred, vid_id, gts):
        tiou = self.tious[0]
        # This method averages the tIoU precision from METEOR, Bleu, etc. across videos
        res = {}
        unique_index = 0

        # video id to unique caption ids mapping
        vid2capid = {}

        cur_res = {}
        cur_gts = {}
        vid2capid = []

        # For each prediction, we look at the tIoU with ground truth.

        has_added = False

        for caption_idx, caption_timestamp in enumerate(gts["timestamps"]):
            if self.iou(pred[1], caption_timestamp) >= tiou:
                cur_res[unique_index] = [{"caption": remove_nonascii(pred[0])}]
                cur_gts[unique_index] = [
                    {"caption": remove_nonascii(gts["sentences"][caption_idx])}
                ]
                vid2capid.append(unique_index)
                unique_index += 1
                has_added = True

        # If the predicted caption does not overlap with any ground truth,
        # we should compare it with garbage.
        if not has_added:
            cur_res[unique_index] = [{"caption": remove_nonascii(pred[0])}]
            cur_gts[unique_index] = [{"caption": random_string(random.randint(10, 20))}]
            vid2capid.append(unique_index)
            unique_index += 1

        # call tokenizer here for all predictions and gts
        tokenize_res = self.tokenizer.tokenize(cur_res)
        tokenize_gts = self.tokenizer.tokenize(cur_gts)
        res = {index: tokenize_res[index] for index in vid2capid}
        gt_temp = {index: tokenize_gts[index] for index in vid2capid}

        # For each video, take all the valid pairs (based from tIoU) and compute the score
        all_scores = []
        stdout_saved = sys.stdout
        sys.stdout = io.StringIO()
        for scorer, method in self.scorers:
            if len(res) == 0 or len(gts) == 0:
                if type(method) == list:
                    score = [0] * len(method)
                else:
                    score = 0
            else:
                score, scores = scorer.compute_score(gt_temp, res)
                score = score[0] if isinstance(score, list) else score
            all_scores.append(score)
        sys.stdout = stdout_saved
        return all_scores, has_added


def calculate_all_scores(
    submission,
    references,
    max_proposals_per_video,
    verbose,
    frames,
    json_results_path,
    delta_t_window, 
):
    """
    Calculate scores for a given submission and references.

    Args:
        submission (str): Path to the submission file.
        references (list): List of paths to the reference files.
        max_proposals_per_video (int): Maximum number of proposals per video.
        verbose (bool): Whether to print verbose output.
        frames (bool): Whether to use frame-level evaluation.
        json_results_path (str): Path to the JSON results file.
        delta_t_window (float): Time window for evaluation.

    Returns:
        None
    """
    try:
        with open(json_results_path, "r") as file_json:
            dict_json = json.load(file_json)
    except FileNotFoundError:
        dict_json = {}
    
    evaluator = ANETcaptions(
        ground_truth_filenames=references,
        prediction_filename=submission,
        max_proposals=max_proposals_per_video,
        verbose=verbose,
        frames=frames,
        delta_t_window=delta_t_window,
    )
    with tqdm(total=len(evaluator.prediction.keys())) as pbar:
        for pred in evaluator.prediction.keys():
            write = False
            if pred not in dict_json.keys():
                dict_json[pred] = {}
            if (
                pred not in evaluator.ground_truths[0]
                and pred not in evaluator.ground_truths[1]
            ):
                continue
            else:
                gts = evaluator.ground_truths[0].get(
                    pred, evaluator.ground_truths[1].get(pred)
                )
            dict_json[pred]["delta_t_window"] = delta_t_window
            conditions = [(False, False), (False, True), (True, False), (True, True)]
            # Results: Normal results
            # Results_h_no_w: Results with history but without weighting
            # Results_w: Results with weighting but without history
            # Results_h: Results with weighting and history
            
            for idx, key in enumerate(
                ["results", "results_h_no_w", "results_w", "results_h"]
            ):
                results = dict_json.get(pred, {}).get(key, {})
                if not results:
                    scores_video = evaluator.evaluate_video(
                        pred,
                        gts,
                        weighted=conditions[idx][0],
                        history=conditions[idx][1],
                    )
                    bleu, meteor, rouge  = zip(*scores_video)
                    results["Bleu_4"] = bleu
                    results["METEOR"] = meteor
                    results["ROUGE_L"] = rouge
                    dict_json[pred][key] = results
                    write = True
                else:
                    print("{}: {} already computed".format(pred, key))
            if write:
                with open(json_results_path, "w") as output_file:
                    json.dump(dict_json, output_file, indent=2)

            pbar.update()


def evaluate_all(frames, delta_t_window):
    """
    Evaluate all frames using the provided parameters.

    Args:
        frames (list): List of frames to evaluate.
        delta_t_window (int): Window of segments to consider in the history mode of the scorer.

    Returns:
        None
    """
    cwd = os.path.abspath(os.path.dirname(__file__))
    max_proposals_per_video = 1000
    verbose = False
    validation_path = os.path.join(cwd, "data/validation")
    captions_data_path = os.path.join(cwd, "data/captions")
    results_path = os.path.join(cwd, "results/json")
    for frame in frames:
        # Change to the new file with results if needed (it must be json file with the format described in the ActivityNet dataset)
        captions = os.path.join(captions_data_path, f"data_{frame}.json")
        # Path to save some additional results provided by our scripts
        json_result_path = os.path.join(results_path, f"results_{frame}.json")
        
        validation = [
            os.path.join(validation_path, f"val_1.json"),
            os.path.join(validation_path, f"val_2.json"),
        ]
        calculate_all_scores(
            captions,
            validation,
            max_proposals_per_video,
            verbose,
            frame,
            json_result_path,
            delta_t_window,
        )


if __name__ == "__main__":
    # Indicate in frames the length for delta t (24,48,72,96,120,150)
    # the script will compute the Live Score metric in all its version (LS, wLS, hLS and whLS).
    # See our paper for more details.
    frames = [150]  # sys.argv[1:]
    delta_t_window = 5  # Temporal window for the analysis of the LVC model in history mode
    
    evaluate_all(frames, delta_t_window)
