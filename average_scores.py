import os
import json

def average_score(result_json_path):
    def get_used_metrics(videos_results):
        """Returns all the metrics used in the result json file.

        Args:
            metrics (list): List of metrics used in the result json file.
        """
        first_key = list(videos_results.keys())[0]
        first_result = videos_results[first_key]
        metrics = list(first_result["results"].keys())
        return metrics
    def get_result_cases(videos_results):
        """Returns all the cases used in the result json file.

        Args:
            videos_results (dict): Dictionary containing the results of the videos.
        """
        first_key = list(videos_results.keys())[0]
        first_result = videos_results[first_key]
        cases = list(first_result.keys())
        cases.remove("delta_t_window")
        return cases
    with open(result_json_path, 'r') as f:
        videos_results = json.load(f)
    metrics = get_used_metrics(videos_results)
    result_cases = get_result_cases(videos_results)
    average_scores = {}
    for result_case in result_cases:
        average_scores[result_case] = {}
        for metric in metrics:
            total_score = 0
            total_videos = len(videos_results)
            avg_results_video = 0
            for key, result in videos_results.items():
                if len(result) == 0:
                    total_videos -= 1
                    continue
                len_captions = len(result[result_case][metric])
                avg_results_video = sum(result[result_case][metric]) / len_captions
                total_score += avg_results_video
                avg_results_video = 0
            average_scores[result_case][metric] = total_score / total_videos
    return average_scores

if __name__ == "__main__":
    # Frames can be 24, 48, 72, 96, 120, 150
    frames = 150
    result_json_path = f"results/json/results_{frames}.json"
    average_scores = average_score(result_json_path)
    print (f"Average scores for {frames}:")
    for result_case, score in average_scores.items():
        print(f"{result_case}: {score}")
    