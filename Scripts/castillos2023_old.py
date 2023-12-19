import h5py
import mne
import numpy as np
from mne import create_info
from mne.io import RawArray
from scipy.io import loadmat
import os.path as osp
import zipfile as z

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


Castillos2023_URL = "https://zenodo.org/records/8255618"


# Each trial contained 12 cycles of a 2.2 second code
NR_CYCLES_PER_TRIAL = 15

# Codes were presented at a 60 Hz monitor refresh rate
PRESENTATION_RATE = 60


class BaseCastillos2023(BaseDataset):
    """c-VEP dataset from Thielen et al. (2021)

    Dataset [1]_ from the study on zero-training c-VEP [2]_.

    .. admonition:: Dataset summary

        =============  =======  =======  ==================  ===============  ===============  ===========
        Name             #Subj    #Chan     #Trials / class  Trials length    Sampling rate      #Sessions
        =============  =======  =======  ==================  ===============  ===============  ===========
        Thielen2021         12       32   18900 NT / 18900 T  0.3s             500Hz                     1
        =============  =======  =======  ==================  ===============  ===============  ===========

    **Dataset description**

    EEG recordings were acquired at a sampling rate of 512 Hz, employing 8 Ag/AgCl electrodes. The Biosemi ActiveTwo EEG
    amplifier was utilized during the experiment. The electrode array consisted of Fz, T7, O1, POz, Oz, Iz, O2, and T8,
    connected as EXG channels. This is a custom electrode montage as optimized in a previous study for c-VEP, see [3]_.

    During the experimental sessions, participants engaged in passive operation (i.e., without feedback) of a 4 x 5
    visual speller brain-computer interface (BCI) comprising 20 distinct classes. Each cell of the symbol grid
    underwent luminance modulation at full contrast, accomplished through pseudo-random noise-codes derived from a
    collection of modulated Gold codes. These codes are binary, have a balanced distribution of ones and zeros, and
    adhere to a limited run-length pattern (maximum run-length of 2 bits). The codes were presented at a presentation
    rate of 60 Hz. As one cycle of these modulated Gold codes contains 126 bits, the duration of one cycle is 2.1
    seconds.

    For each of the five blocks, a trial started with a cueing phase, during which the target symbol was highlighted in
    a green hue for a duration of 1 second. Following this, participants maintained their gaze fixated on the target
    symbol while all symbols flashed in accordance with their respective pseudo-random noise-codes for a duration of
    31.5 seconds (i.e., 15 code cycles). Each block encompassed 20 trials, presented in a randomized sequence, thereby
    ensuring that each symbol was attended to once within the span of a block.

    Note, here, we only load the offline data of this study and ignore the online phase.

    References
    ----------

    .. [1] Thielen, J. (Jordy), Pieter Marsman, Jason Farquhar, Desain, P.W.M. (Peter) (2023): From full calibration to
           zero training for a code-modulated visual evoked potentials brain computer interface. Version 3. Radboud
           University. (dataset).
           DOI: https://doi.org/10.34973/9txv-z787

    .. [2] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
           code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
           056007.
           DOI: https://doi.org/10.1088/1741-2552/abecef

    .. [3] Ahmadi, S., Borhanazad, M., Tump, D., Farquhar, J., & Desain, P. (2019). Low channel count montages using
           sensor tying for VEP-based BCI. Journal of Neural Engineering, 16(6), 066038.
           DOI: https://doi.org/10.1088/1741-2552/ab4057

    Notes
    -----

    .. versionadded:: 0.6.0

    """

    def __init__(self, events, sessions_per_subject, code, paradigm,paradigm_type):
        super().__init__(
            subjects=list(range(1, 12 + 1)),
            sessions_per_subject=sessions_per_subject,
            events=events,
            code=code,
            interval=(0, 0.25),
            paradigm=paradigm,
            doi="https://doi.org/10.1016/j.neuroimage.2023.120446",
        )
        self.paradigm_type = paradigm_type

    def _add_stim_channel_trial(
        self, raw, onsets, labels, offset=200, ch_name="stim_trial"
    ):
        """
        Add a stimulus channel with trial onsets and their labels.

        Parameters
        ----------
        raw: mne.Raw
            The raw object to add the stimulus channel to.
        onsets: List | np.ndarray
            The onsets of the trials in sample numbers.
        labels: List | np.ndarray
            The labels of the trials.
        offset: int (default: 200)
            The integer value to start markers with. For instance, if 200, then label 0 will be marker 200, label 1
            will be be marker 201, etc.
        ch_name: str (default: "stim_trial")
            The name of the added stimulus channel.
        Returns
        -------
        mne.Raw
            The raw object with the added stimulus channel.
        """
        stim_chan = np.zeros((1, len(raw)))
        for onset, label in zip(onsets, labels):
            stim_chan[0, onset] = offset + label
        info = create_info(
            ch_names=["stim_trial"],
            ch_types=["stim"],
            sfreq=raw.info["sfreq"],
            verbose=False,
        )
        raw = raw.add_channels([RawArray(data=stim_chan, info=info, verbose=False)])
        return raw

    def _add_stim_channel_epoch(
        self,
        raw,
        onsets,
        labels,
        codes,
        presentation_rate=60,
        offset=100,
        ch_name="stim_epoch",
    ):
        """
        Add a stimulus channel with epoch onsets and their labels, which are the values of the presented code for each
        of the trials.

        Parameters
        ----------
        raw: mne.Raw
            The raw object to add the stimulus channel to.
        onsets: List | np.ndarray
            The onsets of the trials in sample numbers.
        labels: List | np.ndarray
            The labels of the trials.
        codes: np.ndarray
            The codebook containing each presented code of shape (nr_bits, nr_codes), sampled at the presentation rate.
        presentation_rate: int (default: 60):
            The presentation rate (e.g., frame rate) at which the codes were presented in Hz.
        offset: int (default: 100)
            The integer value to start markers with. For instance, if 100, then label 0 will be marker 100, label 1
            will be be marker 101, etc.
        ch_name: str (default: "stim_epoch")
            The name of the added stimulus channel.
        Returns
        -------
        mne.Raw
            The raw object with the added stimulus channel.
        """
        stim_chan = np.zeros((1, len(raw)))
        for onset, label in zip(onsets, labels):
            idx = np.round(
                onset + np.arange(codes.shape[0]) / presentation_rate * raw.info["sfreq"]
            ).astype("int")
            stim_chan[0, idx] = offset + codes[:, label]
        info = create_info(
            ch_names=[ch_name],
            ch_types=["stim"],
            sfreq=raw.info["sfreq"],
            verbose=False,
        )
        raw = raw.add_channels([RawArray(data=stim_chan, info=info, verbose=False)])
        return raw

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        file_path_list = self.data_path(subject,self.paradigm_type)

        # Codes
        #################### METTRE LES CODE#########################
        
        raw = mne.io.read_raw_eeglab(file_path_list[0],preload=True, verbose=False)

        # There is only one session, one trial of 60 subtrials
        sessions = {"0": {}}
        sessions["0"]["0"] = raw

        # for i_b in range(NR_BLOCKS):
        #     # EEG
        #     raw = mne.io.read_raw_gdf(
        #         file_path_list[2 * i_b],
        #         stim_channel="status",
        #         preload=True,
        #         verbose=False,
        #     )

        #     # Labels at trial level (i.e., symbols)
        #     trial_labels = (
        #         np.array(h5py.File(file_path_list[2 * i_b + 1], "r")["v"])
        #         .astype("uint8")
        #         .flatten()
        #         - 1
        #     )

        #     # Find onsets of trials
        #     # Note, every 2.1 seconds an event was generated: 15 times per trial, plus one 16th "leaking epoch". This
        #     # "leaking epoch" is not always present, so taking epoch[::16, :] won't work.
        #     events = mne.find_events(raw, verbose=False)
        #     cond = np.logical_or(
        #         np.diff(events[:, 0]) < 1.8 * raw.info["sfreq"],
        #         np.diff(events[:, 0]) > 2.4 * raw.info["sfreq"],
        #     )
        #     idx = np.concatenate(([0], 1 + np.where(cond)[0]))
        #     trial_onsets = events[idx, 0]

        #     # Create stim channel with trial information (i.e., symbols)
        #     # Specifically: 200 = symbol-0, 201 = symbol-1, 202 = symbol-2, etc.
        #     raw = self._add_stim_channel_trial(
        #         raw, trial_onsets, trial_labels, offset=200
        #     )

        #     # Create stim channel with epoch information (i.e., 1 / 0, or on / off)
        #     # Specifically: 100 = "0", 101 = "1"
        #     raw = self._add_stim_channel_epoch(
        #         raw, trial_onsets, trial_labels, codes, PRESENTATION_RATE, offset=100
        #     )

        #     # Add data as a new run
        #     run_name = str(i_b)
        #     sessions["0"][run_name] = raw

        return sessions
    
    
    def data_path(
        self, subject, paradigm_type, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of a single subject."""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        sub = f"P{subject:2d}"
        subject_paths = []

        # Channel locations
        # url = "{:s}/P{:d}/P{:d}_{:s}.set".format(Castillos2023_URL, subject, subject, paradigm_type)
        # subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

        url = "https://zenodo.org/records/8255618/files/4Class-CVEP.zip"
        path_zip = dl.data_dl(url, "4Class-VEP",path="C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD")
        path_folder = "C"+path_zip.strip("4Class-VEP.zip")
        print("\n\n\n\n")
        print("the path to the zip file is", path_folder)
        print("the path to the zip file is", path_zip)
        # check if has to unzip
        if not (osp.isdir(path_folder + "4Class-VEP")):
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        subject_paths.append(path_folder+"4Class-CVEP/P{:d}/P{:d}_{:s}.set".format(subject, subject, paradigm_type))

        return subject_paths


class CasitllosBurstVEP100(BaseCastillos2023):
    """SSVEP MAMEM 1 dataset.

    .. admonition:: Dataset summary


        ======  =======  =======  ==========  =================  ===============  ===============  ===========
        Name      #Subj    #Chan    #Classes  #Trials / class    Trials length    Sampling rate      #Sessions
        ======  =======  =======  ==========  =================  ===============  ===============  ===========
        MAMEM1       12      32            4  15                 2.2s               500Hz                    1
        ======  =======  =======  ==========  =================  ===============  ===============  ===========

    """

    def __init__(self):
        super().__init__(
            events={'011110000000000000000000000000111100000000000000000000000011110000000000000000000001111000000000000000000000011110000000000000000011110000000000000000000111100000000000000111100000011110000000000000000000000000111100000000000000000000000011110000000000000000000001111000000000000000000000011110000000000000000011110000000000000000000111100000000000000111100000_1': 1,
                    '000000000000000000000001111000000000000000000011110000000000000011110000000000000111100000000000000001111000000000000000000011110000000000000000000000111100000000000000000011110000000000000000000000000001111000000000000000000011110000000000000011110000000000000111100000000000000001111000000000000000000011110000000000000000000000111100000000000000000011110000_2': 2,
                    '000000000000000000111100000000000000000000111100000000000001111000000000000000000111100000000000000000000000011110000000000000000000000001111000000000000000000000000111100000000000000000000000000000111100000000000000000000111100000000000001111000000000000000000111100000000000000000000000011110000000000000000000000001111000000000000000000000000111100000000000_3': 3,
                    '000000000000000111100000000000000000000001111000000000000000000000000111100000000000000111100000000000000000000001111000000000001111000000000000000000001111000000000000000000000111000000000000000111100000000000000000000001111000000000000000000000000111100000000000000111100000000000000000000001111000000000001111000000000000000000001111000000000000000000000111_4': 4},
            sessions_per_subject=1,
            code="CasitllosBurstVEP100",
            paradigm="burstVEP",
            paradigm_type="burst100",
        )

class CasitllosBurstVEP40(BaseCastillos2023):
    """SSVEP MAMEM 1 dataset.

    .. admonition:: Dataset summary


        ======  =======  =======  ==========  =================  ===============  ===============  ===========
        Name      #Subj    #Chan    #Classes  #Trials / class    Trials length    Sampling rate      #Sessions
        ======  =======  =======  ==========  =================  ===============  ===============  ===========
        MAMEM1       12      32            4  15                 2.2s               500Hz                    1
        ======  =======  =======  ==========  =================  ===============  ===============  ===========

    """

    def __init__(self):
        super().__init__(
            events={'011110000000000000000000000000111100000000000000000000000011110000000000000000000001111000000000000000000000011110000000000000000011110000000000000000000111100000000000000111100000011110000000000000000000000000111100000000000000000000000011110000000000000000000001111000000000000000000000011110000000000000000011110000000000000000000111100000000000000111100000_1': 1,
                    '000000000000000000000001111000000000000000000011110000000000000011110000000000000111100000000000000001111000000000000000000011110000000000000000000000111100000000000000000011110000000000000000000000000001111000000000000000000011110000000000000011110000000000000111100000000000000001111000000000000000000011110000000000000000000000111100000000000000000011110000_2': 2,
                    '000000000000000000111100000000000000000000111100000000000001111000000000000000000111100000000000000000000000011110000000000000000000000001111000000000000000000000000111100000000000000000000000000000111100000000000000000000111100000000000001111000000000000000000111100000000000000000000000011110000000000000000000000001111000000000000000000000000111100000000000_3': 3,
                    '000000000000000111100000000000000000000001111000000000000000000000000111100000000000000111100000000000000000000001111000000000001111000000000000000000001111000000000000000000000111000000000000000111100000000000000000000001111000000000000000000000000111100000000000000111100000000000000000000001111000000000001111000000000000000000001111000000000000000000000111_4': 4},
            sessions_per_subject=1,
            code="CasitllosBurstVEP40",
            paradigm="burstVEP",
            paradigm_type="burst40",
        )

class CasitllosCVEP100(BaseCastillos2023):
    """SSVEP MAMEM 1 dataset.

    .. admonition:: Dataset summary


        ======  =======  =======  ==========  =================  ===============  ===============  ===========
        Name      #Subj    #Chan    #Classes  #Trials / class    Trials length    Sampling rate      #Sessions
        ======  =======  =======  ==========  =================  ===============  ===============  ===========
        MAMEM1       12      32            4  15                 2.2s               500Hz                    1
        ======  =======  =======  ==========  =================  ===============  ===============  ===========

    """

    def __init__(self):
        super().__init__(
            events={'111111111111110000111100001111000011001111001100001100111111000000110011111100001100110000001111001100000011001100001100000000001100_1': 1,
                    '000011110000000011111111110000000000001111000011001100110011110011000000110000001111110011001111111111000011000000111100001100111111_2': 2,
                    '111100000000110000111100000000000011001111001100110011000000111111001111110011001111000000111100111111000000000011111100001100110011_3': 3,
                    '111100111100111100111100000011111100000011111111110000110011000011110000000011000000001111111111110011001100001111000011000000110011_4': 4},
            sessions_per_subject=1,
            code="CasitllosBurstVEP100",
            paradigm="cvep",
            paradigm_type="mseq100",
        )

class CasitllosCVEP40(BaseCastillos2023):
    """SSVEP MAMEM 1 dataset.

    .. admonition:: Dataset summary


        ======  =======  =======  ==========  =================  ===============  ===============  ===========
        Name      #Subj    #Chan    #Classes  #Trials / class    Trials length    Sampling rate      #Sessions
        ======  =======  =======  ==========  =================  ===============  ===============  ===========
        MAMEM1       12      32            4  15                 2.2s               500Hz                    1
        ======  =======  =======  ==========  =================  ===============  ===============  ===========

    """

    def __init__(self):
        super().__init__(
            events={'111111111111110000111100001111000011001111001100001100111111000000110011111100001100110000001111001100000011001100001100000000001100_1': 1,
                    '000011110000000011111111110000000000001111000011001100110011110011000000110000001111110011001111111111000011000000111100001100111111_2': 2,
                    '111100000000110000111100000000000011001111001100110011000000111111001111110011001111000000111100111111000000000011111100001100110011_3': 3,
                    '111100111100111100111100000011111100000011111111110000110011000011110000000011000000001111111111110011001100001111000011000000110011_4': 4},
            sessions_per_subject=1,
            code="CasitllosBurstVEP40",
            paradigm="cvep",
            paradigm_type="mseq40",
        )