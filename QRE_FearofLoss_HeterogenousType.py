# Combined QRE Simulation and MLE Framework
# This script integrates QRE simulation with profit calculation and Maximum Likelihood Estimation (MLE).

import numpy as np
import pandas as pd
import xlsxwriter
from numpy.random import choice
from concurrent.futures import ProcessPoolExecutor

# === Configurations ===
class Config:
    num_slices = 33
    data_points_per_slice = 400000
    beta = 30  # Noise parameter for shocks
    max_effort = 10  # Maximum effort participants can allocate
    reward = 30  # Reward for contest winners
    theta = 20  # Effort impact weight
    effort_step = max_effort / (num_slices - 1)
    efforts1 = np.linspace(0, max_effort, num_slices)
    efforts2 = np.linspace(0, max_effort, num_slices)

# === Input Parameters ===
class Parameters:
    mu_type1 = [16]
    mu_type2 = [0.9]
    fear_loss_solver1_contest1 = [0]
    fear_loss_solver1_contest2 = [0]
    fear_loss_solver2_contest1 = [ 25]
    fear_loss_solver2_contest2 = [117]
    mix_ratios = [0.37]

# === Grid Setup ===
def setup_effort_grid(config):
    effort_combinations = np.arange(1, config.num_slices * config.num_slices + 1)
    e1_indices = ((effort_combinations - 1) / config.num_slices).astype(int)
    e2_indices = (effort_combinations - 1) % config.num_slices

    mask_valid = config.effort_step * (e1_indices + e2_indices) <= config.max_effort
    valid_indices = effort_combinations[mask_valid] - 1

    return valid_indices

# === QRE Computation ===
def compute_qre_probabilities(args, valid_indices, config):
    fear_loss_s1_c1, fear_loss_s1_c2, fear_loss_s2_c1, fear_loss_s2_c2, mu1, mu2, mix_ratio = args

    probabilities = np.zeros(config.num_slices * config.num_slices)
    probabilities[valid_indices] = 1 / len(valid_indices)

    for _ in range(2):
        type1_probabilities = np.zeros_like(probabilities)
        type2_probabilities = np.zeros_like(probabilities)

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(compare_performance, [
                (i, probabilities, fear_loss_s1_c1, fear_loss_s1_c2, fear_loss_s2_c1, fear_loss_s2_c2, config)
                for i in valid_indices
            ]))

        for idx, result in results:
            util1, util2 = result
            type1_probabilities[idx] = util1
            type2_probabilities[idx] = util2

        type1_probabilities = np.exp(type1_probabilities / mu1)
        type2_probabilities = np.exp(type2_probabilities / mu2)

        type1_probabilities /= type1_probabilities.sum()
        type2_probabilities /= type2_probabilities.sum()

        probabilities = mix_ratio * type1_probabilities + (1 - mix_ratio) * type2_probabilities

    return probabilities

# === Performance Comparison ===
def compare_performance(args):
    index, probabilities, fear_loss_s1_c1, fear_loss_s1_c2, fear_loss_s2_c1, fear_loss_s2_c2, config = args
    e1 = index // config.num_slices
    e2 = index % config.num_slices

    shock1 = np.random.uniform(-config.beta, config.beta, config.data_points_per_slice)
    shock2 = np.random.uniform(-config.beta, config.beta, config.data_points_per_slice)

    performance1 = config.theta * np.log(e1 + 1) + shock1
    performance2 = config.theta * np.log(e2 + 1) + shock2

    prob_type1 = (performance1 > performance2).mean()
    prob_type2 = 1 - prob_type1

    utility1 = config.reward * prob_type1 - fear_loss_s1_c1 * (1 - prob_type1) - fear_loss_s1_c2
    utility2 = config.reward * prob_type2 - fear_loss_s2_c1 * (1 - prob_type2) - fear_loss_s2_c2

    return index, (utility1, utility2)

def profit_monte_carlo(probabilities, config):
    """
    Simulate the profit calculation using Monte Carlo methods based on given probabilities.

    Args:
        probabilities (np.ndarray): QRE probabilities.
        config (Config): Configuration parameters.

    Returns:
        float: Total profit across two contests.
    """
    num_points = 250000

    # Generate random shocks for each solver and contest
    shocks = {
        f"shock_{solver}_{contest}": np.random.uniform(-config.beta, config.beta, num_points)
        for solver in range(4)
        for contest in range(2)  # Two contests (contest 0 and contest 1)
    }

    # Sample efforts based on probabilities
    efforts_raw = {
        f"effort_{solver}": choice(range(len(probabilities)), num_points, p=probabilities)
        for solver in range(4)
    }

    # Map efforts to slices for each solver and contest
    efforts = {
        f"effort_{solver}_{contest}": config.effort_step * (
            (efforts_raw[f"effort_{solver}"] - 1) % config.num_slices if contest == 1 else
            (efforts_raw[f"effort_{solver}"] - 1) // config.num_slices
        )
        for solver in range(4)
        for contest in range(2)
    }

    # Initialize performance dictionaries
    performances = {
        f"performance_{solver}_{contest}": np.zeros(num_points)
        for solver in range(4)
        for contest in range(2)
    }

    for key in performances:
        mask = efforts[key.replace("performance", "effort")] > 0
        solver, contest = map(int, key.split("_")[1:])
        performances[key][mask] = (
            shocks[f"shock_{solver}_{contest}"][mask]
            + config.theta * np.log(efforts[key.replace("performance", "effort")][mask])
        )

    # Compare performances for both contests
    compare1 = np.maximum.reduce([performances[f"performance_{solver}_0"] for solver in range(4)])
    compare2 = np.maximum.reduce([performances[f"performance_{solver}_1"] for solver in range(4)])

    # Replace zeros with rewards in comparisons
    compare1[compare1 == 0] = config.reward
    compare2[compare2 == 0] = config.reward

    # Calculate profits
    profit1 = (compare1 - config.reward).sum() / num_points
    profit2 = (compare2 - config.reward).sum() / num_points

    return profit1 + profit2


# === Effort Data Processing ===
def process_effort_data(input_path):
    be1 = pd.read_csv(f"{input_path}ContestTreatments_2021-07-20.csv")[48:]
    all_remaining = pd.read_csv(f"{input_path}ContestTreatments_2021-09-10.csv")[48:]
    all_data = pd.concat([be1, all_remaining])

    session_codes = all_data['Session Code'].unique()
    filtered_data = pd.DataFrame(columns=all_data.columns)

    for session in session_codes:
        session_data = all_data[all_data['Session Code'] == session]
        drop_count = int(4 * len(session_data) / 24)
        filtered_data = pd.concat([filtered_data, session_data[drop_count:]])

    breakeven = filtered_data[filtered_data['Beta'] == 36.8]
    low = filtered_data[filtered_data['Beta'] == 30]

    return {
        "breakeven_exclusive": breakeven[breakeven["Non-Exc"] == "Exc"],
        "breakeven_nonexclusive": breakeven[breakeven["Non-Exc"] != "Exc"],
        "low_exclusive": low[low["Non-Exc"] == "Exc"],
        "low_nonexclusive": low[low["Non-Exc"] != "Exc"],
    }

# === Maximum Likelihood Estimation ===
def non_exclusive_mle(effort_list, problist, config):
    likelihood_results = []

    for entry in problist:
        probabilities = entry[1]
        profit = profit_monte_carlo(probabilities, config) / 2

        expected_effort = 0
        for i in range(config.num_slices):
            for j in range(config.num_slices):
                expected_effort += probabilities[i * config.num_slices + j] * (
                    config.efforts1[i] + config.efforts2[j]
                )

        likelihood_results.append([
            *entry[0],
            expected_effort,
            profit
        ])

    return likelihood_results

# === Export Results ===
def export_results(likelihood_list, output_path, filename):
    workbook = xlsxwriter.Workbook(f"{output_path}{filename}")
    worksheet = workbook.add_worksheet("results")

    headers = [
        "v1", "v2", "l1", "l2", "Mu1", "Mu2", "alfa", "Expected Effort", "Profit"
    ]
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)

    for row, data in enumerate(likelihood_list, start=1):
        for col, value in enumerate(data):
            worksheet.write(row, col, value)

    workbook.close()

# === Main Execution ===
def main():
    config = Config()
    input_path = "./"
    output_path = "./"

    # Process effort data
    effort_data = process_effort_data(input_path)

    # Generate QRE Probabilities
    problist = []
    valid_indices = setup_effort_grid(config)

    for mu1 in Parameters.mu_type1:
        for mu2 in Parameters.mu_type2:
            for fear1_c1 in Parameters.fear_loss_solver1_contest1:
                for fear1_c2 in Parameters.fear_loss_solver1_contest2:
                    for fear2_c1 in Parameters.fear_loss_solver2_contest1:
                        for fear2_c2 in Parameters.fear_loss_solver2_contest2:
                            for mix_ratio in Parameters.mix_ratios:
                                args = (fear1_c1, fear1_c2, fear2_c1, fear2_c2, mu1, mu2, mix_ratio)
                                probabilities = compute_qre_probabilities(args, valid_indices, config)

                                # Ensure probabilities cover the full grid
                                if probabilities.shape[0] != config.num_slices ** 2:
                                    padded_probs = np.zeros(config.num_slices ** 2)
                                    padded_probs[valid_indices] = probabilities
                                    probabilities = padded_probs

                                # Normalize probabilities
                                probabilities /= np.sum(probabilities)

                                problist.append((args, probabilities))

    # effort list for MLE
    effort_list = effort_data["breakeven_nonexclusive"].groupby(["Effort", "Effort2"]).size().reset_index()

    # Perform MLE
    likelihood_results = non_exclusive_mle(effort_list, problist, config)

    # Export results
    export_results(likelihood_results, output_path, "MLE_Results.xlsx")

# Execute main function
if __name__ == "__main__":
    main()

