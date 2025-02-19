import os
from pathlib import Path
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

def run_classifiers(session_folder, id, date):
    session_folder = Path(session_folder)

    # Create new folder for classifiers
    classifiers_folder = session_folder / 'classifiers'
    classifiers_folder.mkdir(parents=True, exist_ok=True)
    
    hearing_loss = 'NIHL' in str(classifiers_folder).lower()
    ontask = 'passive' not in str(classifiers_folder).lower()

    session_name = f"{id} {date}"
    #print(f'Running classifiers for session {session_name}')

    # Load data
    data_files = {}
    for unit_folder in session_folder.glob('raw_metrics/unit_*'):
        for cluster_file in unit_folder.glob('cluster_*_data.csv'):
            cluster_data = pd.read_csv(cluster_file)
            cluster_name = cluster_file.stem.split('_')[1]
            data_files[cluster_name] = cluster_data  # data_files['x'] = cluster_x_dataframe

    # Loop through each unit
    for unit_id, data in tqdm(data_files.items(), desc="Processing clusters", total=len(data_files), leave=True):
        # Create new folder for unit classifiers
        unit_classifiers_folder = classifiers_folder / f'unit_{unit_id}'
        unit_classifiers_folder.mkdir(parents=True, exist_ok=True)

        # Remove trials with NaN spike times
        data = data.dropna(subset=['spike_times'])

        # Run classifiers
        # SPIKE COUNT
        sc_slowfast_slope, sc_slowfast_thresh, sc_slowfast_lapse, sc_slowfast_acc, sc_slowfast_cm, sc_slowfast_pright = spikecount_slowfast_classifier(
            data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=ontask)
        sc_amrate_acc, sc_amrate_cm = spikecount_amrate_classifier(
            data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=ontask)
        if ontask:
            sc_score_acc, sc_score_cm = spikecount_score_classifier(
                data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=ontask)
            sc_choice_acc, sc_choice_cm = spikecount_choice_classifier(
                data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=ontask)
        else:
            sc_score_acc, sc_score_cm = np.nan, np.nan
            sc_choice_acc, sc_choice_cm = np.nan, np.nan

        # PSTH
        psth_slowfast_slope, psth_slowfast_thresh, psth_slowfast_lapse, psth_slowfast_acc, psth_slowfast_cm, psth_slowfast_pright = psth_slowfast_classifier(
            data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=ontask)
        psth_amrate_acc, psth_amrate_cm = psth_amrate_classifier(
            data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=ontask)
        if ontask:
            psth_score_acc, psth_score_cm = psth_score_classifier(
                data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=ontask)
            psth_choice_acc, psth_choice_cm = psth_choice_classifier(
                data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=ontask)
        else:
            psth_score_acc, psth_score_cm = np.nan, np.nan
            psth_choice_acc, psth_choice_cm = np.nan, np.nan

        # Save gaussian fits to csv
        fitted_classifiers = pd.DataFrame(columns=['classifier', 'slope', 'threshold', 'lapse_rate', 'pRight'])
        fitted_classifiers.loc[0] = ['spikecount_slowfast', sc_slowfast_slope, sc_slowfast_thresh, sc_slowfast_lapse, sc_slowfast_pright]
        fitted_classifiers.loc[1] = ['psth_slowfast', psth_slowfast_slope, psth_slowfast_thresh, psth_slowfast_lapse, psth_slowfast_pright]
        if ontask:
            behavior_slope, behavior_thresh, behavior_lapse, behavior_pRight = plot_behavior(data, hearing_loss)
            plt.close()
            fitted_classifiers.loc[2] = ['behavior', behavior_slope, behavior_thresh, behavior_lapse, behavior_pRight]
        fitted_classifiers.to_csv(unit_classifiers_folder / f'unit_{unit_id}_slowfast_classifier_fits.csv', index=False)

        # Save classifier accuracies and confusion matrices to csv
        classifier_accuracies = pd.DataFrame(columns=['classifier', 'accuracy', 'confusion_matrix'])
        classifier_accuracies.loc[0] = ['spikecount_slowfast', sc_slowfast_acc, sc_slowfast_cm]
        classifier_accuracies.loc[1] = ['spikecount_amrate', sc_amrate_acc, sc_amrate_cm]
        classifier_accuracies.loc[2] = ['spikecount_score', sc_score_acc, sc_score_cm]
        classifier_accuracies.loc[3] = ['spikecount_choice', sc_choice_acc, sc_choice_cm]
        classifier_accuracies.loc[4] = ['psth_slowfast', psth_slowfast_acc, psth_slowfast_cm]
        classifier_accuracies.loc[5] = ['psth_amrate', psth_amrate_acc, psth_amrate_cm]
        classifier_accuracies.loc[6] = ['psth_score', psth_score_acc, psth_score_cm]
        classifier_accuracies.loc[7] = ['psth_choice', psth_choice_acc, psth_choice_cm]
        classifier_accuracies.to_csv(unit_classifiers_folder / f'unit_{unit_id}_classifier_perfomances.csv', index=False)

    return session_folder, id, date

##### HELPER FUNCTIONS #####
def get_spike_count(Spikes):
    N = len(Spikes)
    start = 200
    stop = 1000
    sc = np.full(N, np.nan)
    
    for i in range(N):
        # Clean the spike times string: remove newlines, extra spaces, and split by commas
        temp = Spikes[i].replace('[', '').replace(']', '').replace('\n', '').split()
        # Convert cleaned strings to float
        tmp = np.array(temp, dtype=float)        
        # Select spikes within the specified range (start to stop)
        sel = (tmp >= start) & (tmp <= stop)        
        # Count the selected spikes
        sc[i] = np.sum(sel)
    
    return sc

def get_psth(Spikes):
    N = len(Spikes)
    bw = 10
    start = 200
    stop = 1200
    bins = np.arange(start, stop + bw, bw)
    nbins = len(bins) - 1  # nbins should exclude the final bin edge
    psth = np.zeros((N, nbins))  # Initialize with zeros instead of NaN
    
    for i in range(N):
        temp = Spikes[i].replace('[', '').replace(']', '').replace('\n', '').split()
        if temp:  # Ensure that temp is not empty
            tmp = np.array(temp, dtype=float)
            sel = (tmp >= start) & (tmp <= stop)
            tmp = tmp[sel]
            psth[i, :] = np.histogram(tmp, bins=bins)[0]
    
    return psth

def fit_gaussian(x, y):
    mu = 6.25
    sigma = 0.50
    xfit = np.linspace(min(x), max(x), 100)
    
    # Define the normal CDF model
    def ncdf(x, mu, sigma):
        return norm.cdf(x, loc=mu, scale=sigma)
    
    # Fit the Gaussian model using scipy's curve fitting
    popt, _ = curve_fit(ncdf, x, y, p0=[mu, sigma], maxfev=10000)
    
    # Get the slope and threshold
    threshold = popt[0]
    slope = 1 / popt[1]
    
    # Calculate the fitted y values
    yfit = ncdf(xfit, *popt)
    
    return slope, threshold, xfit, yfit

def plot_behavior(data, hearing_loss):
    if hearing_loss:
        plot_color = 'orange'
    else:
        plot_color = 'black'
    all_rates = np.sort(data['AM_rate'].unique())
    pRight = np.full(len(all_rates), np.nan)
    for i, rate in enumerate(all_rates):
        pRight[i] = np.mean(data[data['AM_rate'] == rate]['choice'] == 'Right')
    slope, threshold, xfit, yfit = behavior_curve = fit_gaussian(all_rates, pRight)
    lapse_rate = (yfit[0] + (1-yfit[-1]))/2
    plt.plot(xfit, yfit, color=plot_color, label='Behavior Performance')
    plt.scatter(all_rates, pRight, color=plot_color)
    return slope, threshold, lapse_rate, pRight

##### SPIKE COUNT CLASSIFIERS #####

def spikecount_slowfast_classifier(data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=True):

    # Find correct trials only if on-task
    '''if ontask:
        corr_trials = data[data['score'] == 'Correct']
    else:
        corr_trials = data'''
    corr_trials = data

    iterations = 1000
    AMrate = corr_trials['AM_rate'].values
    idxAM = np.arange(len(AMrate))
    Spikes = corr_trials['spike_times'].values
    Spikes = get_spike_count(Spikes).reshape(-1, 1)
    
    uRate = np.sort(np.unique(AMrate))
    Nrates = len(uRate)
    Comp = np.full((iterations, Nrates), np.nan)

    for i in range(iterations):
        for j in range(Nrates):
            # Test
            tel = np.where(AMrate == uRate[j])[0]
            AMtemp = Spikes[tel, :]
            ntrials = AMtemp.shape[0]
            idx = round(ntrials * 0.20)
            idxvec = np.random.permutation(ntrials)[:idx]
            amtest = np.nanmean(AMtemp[idxvec, :], axis=0)

            AMrest = np.setdiff1d(idxAM, tel)
            am = np.setdiff1d(tel, idxvec)
            AM = np.concatenate((am, AMrest))
            
            # Templates
            TempSpikes = Spikes[AM, :]

            # Slow AM Rates
            sel = AMrate[AM] < uRate[4]
            slow = TempSpikes[sel, :]
            nslowtrials = slow.shape[0]
            idx = round(nslowtrials * 0.80)
            tempslow = np.random.permutation(nslowtrials)[:idx]
            TempSlow = np.nanmean(slow[tempslow, :], axis=0)

            # Fast AM Rates
            zel = AMrate[AM] > uRate[4]
            fast = TempSpikes[zel, :]
            nfasttrials = fast.shape[0]
            idx = round(nfasttrials * 0.80)
            tempfast = np.random.permutation(nfasttrials)[:idx]
            TempFast = np.nanmean(fast[tempfast, :], axis=0)

            # Compare with templates
            SlowComp = np.abs(amtest - TempSlow)
            FastComp = np.abs(amtest - TempFast)
            Cmp = [SlowComp, FastComp]
            index = np.argmin(Cmp)
            Comp[i, j] = 1 if index == 1 else 0

    y = np.mean(Comp, axis=0)

    # Generate confusion matrix
    confusion_matrix = np.zeros((Nrates, 2))
    for j in range(Nrates):
        confusion_matrix[j, 0] = np.mean(Comp[:, j] == 0)  # Proportion classified as slow
        confusion_matrix[j, 1] = np.mean(Comp[:, j] == 1)  # Proportion classified as fast

    # Normalize confusion matrix so each row sums to 1
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    # Calculate accuracy
    slow_rates = uRate < 6.25
    fast_rates = uRate > 6.25

    slow_accuracy = np.mean(confusion_matrix[slow_rates, 0])
    fast_accuracy = np.mean(confusion_matrix[fast_rates, 1])

    accuracy = (slow_accuracy + fast_accuracy) / 2 * 100

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Slow', 'Fast'])
    ax.set_yticks(np.arange(Nrates))
    ax.set_yticklabels([f'{rate:.2f}' for rate in uRate])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.suptitle(f'Overall Accuracy: {accuracy:.2f}%')
    plt.title(f'{session_name} - Unit {unit_id} - Slow vs. Fast SC Classifier')
    plt.tight_layout()
    figname = unit_classifiers_folder / f'cluster_{unit_id}_spikecount_slowfast_classifier_confusion.pdf'
    plt.savefig(figname, format='pdf')
    plt.close()
    
    # Fit Gaussian model
    slope, threshold, xfit, yfit = fit_gaussian(uRate, y)
    
    # Plot results
    plt.plot(xfit, yfit, 'green', label='Spike Count Classifier Performance')
    plt.scatter(uRate, y, color='green')
    plt.xlabel('AM Rate (Hz)')
    plt.ylabel('Prop. Choose Right')
    plt.xticks(uRate, uRate)
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.xscale('log')
    plt.xlim([3.99, 10])
    plt.ylim([0, 1])
    
    lapse_rate = (y[0] + (1-y[-1]))/2

    # Plot behavior data too if on-task
    if ontask:
        plot_behavior(data, hearing_loss)

    plt.title(f'{session_name} - Unit {unit_id} - Spike Count Slow vs. Fast')
    plt.legend()
    figname = unit_classifiers_folder / f'cluster_{unit_id}_spikecount_slowfast_classifier.pdf'
    plt.savefig(figname, format='pdf')
    plt.close()


    return slope, threshold, lapse_rate, accuracy, confusion_matrix, y

def spikecount_amrate_classifier(data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=True):

    # Find correct trials only if on-task
    '''if ontask:
        corr_trials = data[data['score'] == 'Correct']
    else:
        corr_trials = data'''
    corr_trials = data

    iterations = 1000
    AMrate = corr_trials['AM_rate'].values
    idxAM = np.arange(len(AMrate))
    Spikes = corr_trials['spike_times'].values
    Spikes = get_spike_count(Spikes).reshape(-1, 1)
    
    uRate = np.sort(np.unique(AMrate))
    Nrates = len(uRate)
    Comp = np.full((iterations, Nrates), np.nan)

    for i in range(iterations):
        for j in range(Nrates):
            # Test
            tel = np.where(AMrate == uRate[j])[0]
            AMtemp = Spikes[tel, :]
            ntrials = AMtemp.shape[0]
            idx = round(ntrials * 0.20)
            idxvec = np.random.permutation(ntrials)[:idx]
            amtest = np.nanmean(AMtemp[idxvec, :], axis=0)

            AMrest = np.setdiff1d(idxAM, tel)
            am = np.setdiff1d(tel, idxvec)
            AM = np.concatenate((am, AMrest))
            
            # Templates
            TempSpikes = Spikes[AM, :]

            # Generate templates for each AM rate
            templates = {}
            for rate in uRate:
                sel = AMrate[AM] == rate
                rate_spikes = TempSpikes[sel, :]
                ntrials_rate = rate_spikes.shape[0]
                idx = round(ntrials_rate * 0.80)
                temp_rate = np.random.permutation(ntrials_rate)[:idx]
                templates[rate] = np.nanmean(rate_spikes[temp_rate, :], axis=0)

            # Compare with templates
            comparisons = {rate: np.abs(amtest - template) for rate, template in templates.items()}
            predicted_rate = min(comparisons, key=comparisons.get)
            Comp[i, j] = predicted_rate

    # Generate confusion matrix
    confusion_matrix = np.zeros((Nrates, Nrates))
    for i in range(Nrates):
        for j in range(Nrates):
            confusion_matrix[i, j] = np.mean(Comp[:, i] == uRate[j])

    # Normalize confusion matrix so each row sums to 1
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    # Calculate accuracy
    accuracy = np.mean(np.diag(confusion_matrix)) * 100

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    ax.set_xticks(np.arange(Nrates))
    ax.set_xticklabels([f'{rate:.2f}' for rate in uRate])
    ax.set_yticks(np.arange(Nrates))
    ax.set_yticklabels([f'{rate:.2f}' for rate in uRate])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.suptitle(f'Overall Accuracy: {accuracy:.2f}%')
    plt.title(f'{session_name} - Unit {unit_id} - AM Rate SC Classifier')
    plt.tight_layout()
    figname = unit_classifiers_folder / f'cluster_{unit_id}_spikecount_amrate_classifier_confusion.pdf'
    plt.savefig(figname, format='pdf')
    plt.close()

    return accuracy, confusion_matrix

def spikecount_score_classifier(data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=True):

    # Keep only correct and incorrect trials
    scored_trials = data[data['score'].isin(['Correct', 'Incorrect'])]

    iterations = 1000
    scores = scored_trials['score'].values
    idxScores = np.arange(len(scores))
    Spikes = scored_trials['spike_times'].values
    Spikes = get_spike_count(Spikes).reshape(-1, 1)
    
    uScores = np.array(['Correct', 'Incorrect'])
    Nscores = len(uScores)
    Comp = np.full((iterations, Nscores), np.nan)

    for i in range(iterations):
        for j in range(Nscores):
            # Test
            tel = np.where(scores == uScores[j])[0]
            ScoreTemp = Spikes[tel, :]
            ntrials = ScoreTemp.shape[0]
            idx = round(ntrials * 0.20)
            idxvec = np.random.permutation(ntrials)[:idx]
            scoretest = np.nanmean(ScoreTemp[idxvec, :], axis=0)

            ScoreRest = np.setdiff1d(idxScores, tel)
            score = np.setdiff1d(tel, idxvec)
            Score = np.concatenate((score, ScoreRest))
            
            # Templates
            TempSpikes = Spikes[Score, :]

            # Correct trials
            sel = scores[Score] == 'Correct'
            correct = TempSpikes[sel, :]
            ncorrecttrials = correct.shape[0]
            idx = round(ncorrecttrials * 0.80)
            tempcorrect = np.random.permutation(ncorrecttrials)[:idx]
            TempCorrect = np.nanmean(correct[tempcorrect, :], axis=0)

            # Incorrect trials
            zel = scores[Score] == 'Incorrect'
            incorrect = TempSpikes[zel, :]
            nincorrecttrials = incorrect.shape[0]
            idx = round(nincorrecttrials * 0.80)
            tempincorrect = np.random.permutation(nincorrecttrials)[:idx]
            TempIncorrect = np.nanmean(incorrect[tempincorrect, :], axis=0)

            # Compare with templates
            CorrectComp = np.abs(scoretest - TempCorrect)
            IncorrectComp = np.abs(scoretest - TempIncorrect)
            Cmp = [CorrectComp, IncorrectComp]
            index = np.argmin(Cmp)
            Comp[i, j] = 1 if index == 1 else 0

    y = np.mean(Comp, axis=0)

    # Generate confusion matrix
    confusion_matrix = np.zeros((Nscores, 2))
    for j in range(Nscores):
        confusion_matrix[j, 0] = np.mean(Comp[:, j] == 0)  # Proportion classified as correct
        confusion_matrix[j, 1] = np.mean(Comp[:, j] == 1)  # Proportion classified as incorrect

    # Normalize confusion matrix so each row sums to 1
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    # Calculate accuracy
    correct_accuracy = np.mean(confusion_matrix[0, 0])
    incorrect_accuracy = np.mean(confusion_matrix[1, 1])

    accuracy = (correct_accuracy + incorrect_accuracy) / 2 * 100

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Correct', 'Incorrect'])
    ax.set_yticks(np.arange(Nscores))
    ax.set_yticklabels(['Correct', 'Incorrect'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.suptitle(f'Overall Accuracy: {accuracy:.2f}%')
    plt.title(f'{session_name} - Unit {unit_id} - Score SC Classifier')
    plt.tight_layout()
    figname = unit_classifiers_folder / f'cluster_{unit_id}_spikecount_score_classifier_confusion.pdf'
    plt.savefig(figname, format='pdf')
    plt.close()

    return accuracy, confusion_matrix

def spikecount_choice_classifier(data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=True):

    # Keep only trials with choices
    choice_trials = data[data['choice'].isin(['Left', 'Right'])]

    iterations = 1000
    choices = choice_trials['choice'].values
    idxChoices = np.arange(len(choices))
    Spikes = choice_trials['spike_times'].values
    Spikes = get_spike_count(Spikes).reshape(-1, 1)
    
    uChoices = np.array(['Left', 'Right'])
    Nchoices = len(uChoices)
    Comp = np.full((iterations, Nchoices), np.nan)

    for i in range(iterations):
        for j in range(Nchoices):
            # Test
            tel = np.where(choices == uChoices[j])[0]
            ChoiceTemp = Spikes[tel, :]
            ntrials = ChoiceTemp.shape[0]
            idx = round(ntrials * 0.20)
            idxvec = np.random.permutation(ntrials)[:idx]
            choicetest = np.nanmean(ChoiceTemp[idxvec, :], axis=0)

            ChoiceRest = np.setdiff1d(idxChoices, tel)
            choice = np.setdiff1d(tel, idxvec)
            Choice = np.concatenate((choice, ChoiceRest))
            
            # Templates
            TempSpikes = Spikes[Choice, :]

            # Left choices
            sel = choices[Choice] == 'Left'
            left = TempSpikes[sel, :]
            nlefttrials = left.shape[0]
            idx = round(nlefttrials * 0.80)
            templeft = np.random.permutation(nlefttrials)[:idx]
            TempLeft = np.nanmean(left[templeft, :], axis=0)

            # Right choices
            zel = choices[Choice] == 'Right'
            right = TempSpikes[zel, :]
            nrighttrials = right.shape[0]
            idx = round(nrighttrials * 0.80)
            tempright = np.random.permutation(nrighttrials)[:idx]
            TempRight = np.nanmean(right[tempright, :], axis=0)

            # Compare with templates
            LeftComp = np.abs(choicetest - TempLeft)
            RightComp = np.abs(choicetest - TempRight)
            Cmp = [LeftComp, RightComp]
            index = np.argmin(Cmp)
            Comp[i, j] = 1 if index == 1 else 0

    y = np.mean(Comp, axis=0)

    # Generate confusion matrix
    confusion_matrix = np.zeros((Nchoices, 2))
    for j in range(Nchoices):
        confusion_matrix[j, 0] = np.mean(Comp[:, j] == 0)  # Proportion classified as left
        confusion_matrix[j, 1] = np.mean(Comp[:, j] == 1)  # Proportion classified as right

    # Normalize confusion matrix so each row sums to 1
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    # Calculate accuracy
    left_accuracy = np.mean(confusion_matrix[0, 0])
    right_accuracy = np.mean(confusion_matrix[1, 1])

    accuracy = (left_accuracy + right_accuracy) / 2 * 100

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Left', 'Right'])
    ax.set_yticks(np.arange(Nchoices))
    ax.set_yticklabels(['Left', 'Right'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.suptitle(f'Overall Accuracy: {accuracy:.2f}%')
    plt.title(f'{session_name} - Unit {unit_id} - Choice SC Classifier')
    plt.tight_layout()
    figname = unit_classifiers_folder / f'cluster_{unit_id}_spikecount_choice_classifier_confusion.pdf'
    plt.savefig(figname, format='pdf')
    plt.close()

    return accuracy, confusion_matrix

##### PSTH CLASSIFIERS #####

def psth_slowfast_classifier(data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=True):

    # Find correct trials only if on-task
    '''if ontask:
        corr_trials = data[data['score'] == 'Correct']
    else:
        corr_trials = data'''
    corr_trials = data

    # Remove trials with NaN spike times
    corr_trials = corr_trials.dropna(subset=['spike_times'])

    iterations = 1000
    AMrate = corr_trials['AM_rate'].values
    idxAM = np.arange(len(AMrate))
    Spikes = corr_trials['spike_times'].values
    Spikes = get_psth(Spikes)
    
    uRate = np.unique(AMrate)
    Nrates = len(uRate)
    Comp = np.full((iterations, Nrates), np.nan)

    for i in range(iterations):
        for j in range(Nrates):
            # Test
            tel = np.where(AMrate == uRate[j])[0]
            AMtemp = Spikes[tel, :]
            ntrials = AMtemp.shape[0]
            idx = round(ntrials * 0.20)
            idxvec = np.random.permutation(ntrials)[:idx]
            amtest = np.nanmean(AMtemp[idxvec, :], axis=0)

            AMrest = np.setdiff1d(idxAM, tel)
            am = np.setdiff1d(tel, idxvec)
            AM = np.concatenate((am, AMrest))
            
            # Templates
            TempSpikes = Spikes[AM, :]

            # Slow AM Rates
            sel = AMrate[AM] < uRate[4]
            slow = TempSpikes[sel, :]
            nslowtrials = slow.shape[0]
            idx = round(nslowtrials * 0.80)
            tempslow = np.random.permutation(nslowtrials)[:idx]
            TempSlow = np.nanmean(slow[tempslow, :], axis=0)

            # Fast AM Rates
            zel = AMrate[AM] > uRate[4]
            fast = TempSpikes[zel, :]
            nfasttrials = fast.shape[0]
            idx = round(nfasttrials * 0.80)
            tempfast = np.random.permutation(nfasttrials)[:idx]
            TempFast = np.nanmean(fast[tempfast, :], axis=0)

            # Compare with templates
            SlowComp = np.linalg.norm(amtest - TempSlow)
            FastComp = np.linalg.norm(amtest - TempFast)
            Cmp = [SlowComp, FastComp]
            index = np.argmin(Cmp)
            Comp[i, j] = 1 if index == 1 else 0

    y = np.mean(Comp, axis=0)
    
    # Generate confusion matrix
    confusion_matrix = np.zeros((Nrates, 2))
    for j in range(Nrates):
        confusion_matrix[j, 0] = np.mean(Comp[:, j] == 0)  # Proportion classified as slow
        confusion_matrix[j, 1] = np.mean(Comp[:, j] == 1)  # Proportion classified as fast

    # Normalize confusion matrix so each row sums to 1
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    # Calculate accuracy
    slow_rates = uRate < 6.25
    fast_rates = uRate > 6.25

    slow_accuracy = np.mean(confusion_matrix[slow_rates, 0])
    fast_accuracy = np.mean(confusion_matrix[fast_rates, 1])

    accuracy = (slow_accuracy + fast_accuracy) / 2 * 100

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Slow', 'Fast'])
    ax.set_yticks(np.arange(Nrates))
    ax.set_yticklabels([f'{rate:.2f}' for rate in uRate])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.suptitle(f'Overall Accuracy: {accuracy:.2f}%')
    plt.title(f'{session_name} - Unit {unit_id} - Slow vs. Fast PSTH Classifier')
    plt.tight_layout()
    figname = unit_classifiers_folder / f'cluster_{unit_id}_psth_slowfast_classifier_confusion.pdf'
    plt.savefig(figname, format='pdf')
    plt.close()
    
    # Fit Gaussian model
    slope, threshold, xfit, yfit = fit_gaussian(uRate, y)
    
    # Plot results
    plt.plot(xfit, yfit, 'blue', label='PSTH Classifier Performance')
    plt.scatter(uRate, y, color='blue')
    plt.xlabel('AM Rate (Hz)')
    plt.ylabel('Prop. Choose Right')
    plt.xticks(uRate, uRate)
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.xscale('log')
    plt.xlim([3.99, 10])
    plt.ylim([0, 1])
    
    lapse_rate = (y[0] + (1-y[-1]))/2

    # Plot behavior data too if on-task
    if ontask:
        plot_behavior(data, hearing_loss)
    
    plt.title(f'{session_name} - Unit {unit_id} - PSTH Slow vs. Fast')
    plt.legend()
    figname = unit_classifiers_folder / f'cluster_{unit_id}_psth_slowfast_classifier.pdf'
    plt.savefig(figname, format='pdf')
    plt.close()

    return slope, threshold, lapse_rate, accuracy, confusion_matrix, y

def psth_amrate_classifier(data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=True):

    # Find correct trials only if on-task
    '''if ontask:
        corr_trials = data[data['score'] == 'Correct']
    else:
        corr_trials = data'''
    corr_trials = data

    # Remove trials with NaN spike times
    corr_trials = corr_trials.dropna(subset=['spike_times'])

    iterations = 1000
    AMrate = corr_trials['AM_rate'].values
    idxAM = np.arange(len(AMrate))
    Spikes = corr_trials['spike_times'].values
    Spikes = get_psth(Spikes)
    
    uRate = np.unique(AMrate)
    Nrates = len(uRate)
    Comp = np.full((iterations, Nrates), np.nan)

    for i in range(iterations):
        for j in range(Nrates):
            # Test
            tel = np.where(AMrate == uRate[j])[0]
            AMtemp = Spikes[tel, :]
            ntrials = AMtemp.shape[0]
            idx = round(ntrials * 0.20)
            idxvec = np.random.permutation(ntrials)[:idx]
            amtest = np.nanmean(AMtemp[idxvec, :], axis=0)

            AMrest = np.setdiff1d(idxAM, tel)
            am = np.setdiff1d(tel, idxvec)
            AM = np.concatenate((am, AMrest))
            
            # Templates
            TempSpikes = Spikes[AM, :]

            # Generate templates for each AM rate
            templates = {}
            for rate in uRate:
                sel = AMrate[AM] == rate
                rate_spikes = TempSpikes[sel, :]
                ntrials_rate = rate_spikes.shape[0]
                idx = round(ntrials_rate * 0.80)
                temp_rate = np.random.permutation(ntrials_rate)[:idx]
                templates[rate] = np.nanmean(rate_spikes[temp_rate, :], axis=0)

            # Compare with templates
            comparisons = {rate: np.linalg.norm(amtest - template) for rate, template in templates.items()}
            predicted_rate = min(comparisons, key=comparisons.get)
            Comp[i, j] = predicted_rate

    # Generate confusion matrix
    confusion_matrix = np.zeros((Nrates, Nrates))
    for i in range(Nrates):
        for j in range(Nrates):
            confusion_matrix[i, j] = np.mean(Comp[:, i] == uRate[j])

    # Normalize confusion matrix so each row sums to 1
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    # Calculate accuracy
    accuracy = np.mean(np.diag(confusion_matrix)) * 100

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    ax.set_xticks(np.arange(Nrates))
    ax.set_xticklabels([f'{rate:.2f}' for rate in uRate])
    ax.set_yticks(np.arange(Nrates))
    ax.set_yticklabels([f'{rate:.2f}' for rate in uRate])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.suptitle(f'Overall Accuracy: {accuracy:.2f}%')
    plt.title(f'{session_name} - Unit {unit_id} - AM Rate PSTH Classifier')
    plt.tight_layout()
    figname = unit_classifiers_folder / f'cluster_{unit_id}_psth_amrate_classifier_confusion.pdf'
    plt.savefig(figname, format='pdf')
    plt.close()

    return accuracy, confusion_matrix

def psth_score_classifier(data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=True):

    # Keep only correct and incorrect trials
    scored_trials = data[data['score'].isin(['Correct', 'Incorrect'])]

    # Remove trials with NaN spike times
    scored_trials = scored_trials.dropna(subset=['spike_times'])

    iterations = 1000
    scores = scored_trials['score'].values
    idxScores = np.arange(len(scores))
    Spikes = scored_trials['spike_times'].values
    Spikes = get_psth(Spikes)
    
    uScores = np.array(['Correct', 'Incorrect'])
    Nscores = len(uScores)
    Comp = np.full((iterations, Nscores), np.nan)

    for i in range(iterations):
        for j in range(Nscores):
            # Test
            tel = np.where(scores == uScores[j])[0]
            ScoreTemp = Spikes[tel, :]
            ntrials = ScoreTemp.shape[0]
            idx = round(ntrials * 0.20)
            idxvec = np.random.permutation(ntrials)[:idx]
            scoretest = np.nanmean(ScoreTemp[idxvec, :], axis=0)

            ScoreRest = np.setdiff1d(idxScores, tel)
            score = np.setdiff1d(tel, idxvec)
            Score = np.concatenate((score, ScoreRest))
            
            # Templates
            TempSpikes = Spikes[Score, :]

            # Correct trials
            sel = scores[Score] == 'Correct'
            correct = TempSpikes[sel, :]
            ncorrecttrials = correct.shape[0]
            idx = round(ncorrecttrials * 0.80)
            tempcorrect = np.random.permutation(ncorrecttrials)[:idx]
            TempCorrect = np.nanmean(correct[tempcorrect, :], axis=0)

            # Incorrect trials
            zel = scores[Score] == 'Incorrect'
            incorrect = TempSpikes[zel, :]
            nincorrecttrials = incorrect.shape[0]
            idx = round(nincorrecttrials * 0.80)
            tempincorrect = np.random.permutation(nincorrecttrials)[:idx]
            TempIncorrect = np.nanmean(incorrect[tempincorrect, :], axis=0)

            # Compare with templates
            CorrectComp = np.linalg.norm(scoretest - TempCorrect)
            IncorrectComp = np.linalg.norm(scoretest - TempIncorrect)
            Cmp = [CorrectComp, IncorrectComp]
            index = np.argmin(Cmp)
            Comp[i, j] = 1 if index == 1 else 0

    y = np.mean(Comp, axis=0)

    # Generate confusion matrix
    confusion_matrix = np.zeros((Nscores, 2))
    for j in range(Nscores):
        confusion_matrix[j, 0] = np.mean(Comp[:, j] == 0)  # Proportion classified as correct
        confusion_matrix[j, 1] = np.mean(Comp[:, j] == 1)  # Proportion classified as incorrect

    # Normalize confusion matrix so each row sums to 1
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    # Calculate accuracy
    correct_accuracy = np.mean(confusion_matrix[0, 0])
    incorrect_accuracy = np.mean(confusion_matrix[1, 1])

    accuracy = (correct_accuracy + incorrect_accuracy) / 2 * 100

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Correct', 'Incorrect'])
    ax.set_yticks(np.arange(Nscores))
    ax.set_yticklabels(['Correct', 'Incorrect'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.suptitle(f'Overall Accuracy: {accuracy:.2f}%')
    plt.title(f'{session_name} - Unit {unit_id} - Score PSTH Classifier')
    plt.tight_layout()
    figname = unit_classifiers_folder / f'cluster_{unit_id}_psth_score_classifier_confusion.pdf'
    plt.savefig(figname, format='pdf')
    plt.close()

    return accuracy, confusion_matrix

def psth_choice_classifier(data, hearing_loss, session_name, unit_id, unit_classifiers_folder, ontask=True):

    # Keep only trials with choices
    choice_trials = data[data['choice'].isin(['Left', 'Right'])]

    # Remove trials with NaN spike times
    choice_trials = choice_trials.dropna(subset=['spike_times'])

    iterations = 1000
    choices = choice_trials['choice'].values
    idxChoices = np.arange(len(choices))
    Spikes = choice_trials['spike_times'].values
    Spikes = get_psth(Spikes)
    
    uChoices = np.array(['Left', 'Right'])
    Nchoices = len(uChoices)
    Comp = np.full((iterations, Nchoices), np.nan)

    for i in range(iterations):
        for j in range(Nchoices):
            # Test
            tel = np.where(choices == uChoices[j])[0]
            ChoiceTemp = Spikes[tel, :]
            ntrials = ChoiceTemp.shape[0]
            idx = round(ntrials * 0.20)
            idxvec = np.random.permutation(ntrials)[:idx]
            choicetest = np.nanmean(ChoiceTemp[idxvec, :], axis=0)

            ChoiceRest = np.setdiff1d(idxChoices, tel)
            choice = np.setdiff1d(tel, idxvec)
            Choice = np.concatenate((choice, ChoiceRest))
            
            # Templates
            TempSpikes = Spikes[Choice, :]

            # Left choices
            sel = choices[Choice] == 'Left'
            left = TempSpikes[sel, :]
            nlefttrials = left.shape[0]
            idx = round(nlefttrials * 0.80)
            templeft = np.random.permutation(nlefttrials)[:idx]
            TempLeft = np.nanmean(left[templeft, :], axis=0)

            # Right choices
            zel = choices[Choice] == 'Right'
            right = TempSpikes[zel, :]
            nrighttrials = right.shape[0]
            idx = round(nrighttrials * 0.80)
            tempright = np.random.permutation(nrighttrials)[:idx]
            TempRight = np.nanmean(right[tempright, :], axis=0)

            # Compare with templates
            LeftComp = np.linalg.norm(choicetest - TempLeft)
            RightComp = np.linalg.norm(choicetest - TempRight)
            Cmp = [LeftComp, RightComp]
            index = np.argmin(Cmp)
            Comp[i, j] = 1 if index == 1 else 0

    y = np.mean(Comp, axis=0)

    # Generate confusion matrix
    confusion_matrix = np.zeros((Nchoices, 2))
    for j in range(Nchoices):
        confusion_matrix[j, 0] = np.mean(Comp[:, j] == 0)  # Proportion classified as left
        confusion_matrix[j, 1] = np.mean(Comp[:, j] == 1)  # Proportion classified as right

    # Normalize confusion matrix so each row sums to 1
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    # Calculate accuracy
    left_accuracy = np.mean(confusion_matrix[0, 0])
    right_accuracy = np.mean(confusion_matrix[1, 1])

    accuracy = (left_accuracy + right_accuracy) / 2 * 100

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Left', 'Right'])
    ax.set_yticks(np.arange(Nchoices))
    ax.set_yticklabels(['Left', 'Right'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.suptitle(f'Overall Accuracy: {accuracy:.2f}%')
    plt.title(f'{session_name} - Unit {unit_id} - Choice PSTH Classifier')
    plt.tight_layout()
    figname = unit_classifiers_folder / f'cluster_{unit_id}_psth_choice_classifier_confusion.pdf'
    plt.savefig(figname, format='pdf')
    plt.close()

    return accuracy, confusion_matrix
