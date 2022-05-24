# #################################################################################
# MODIS aerosol optical depth spatial downscaling and post-process correction
# for the DRAGON campaign 2011
# ---
#
# CODE VERSION 24 May 2022.
#
#   * Developed by: Finnish Meteorological Institute and University of Eastern Finland
#   * Contact info: Antti Lipponen (antti.lipponen@fmi.fi)
#
#  To download and prepare the MODIS and AERONET data run downloadprepare.sh
#  script before running this correction script.
#
# #################################################################################
import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


if __name__ == "__main__":

    if not os.path.isdir(os.path.join('data', 'preparedData')) or not os.path.isfile(os.path.join('data', 'DRAGON2011_AERONETdf.h5')) or not os.path.isfile(os.path.join('data', 'DRAGON2011_MODISdf.h5')):
        print('ERROR: Data not found. You first need to run the downloadprepare.sh script to download and prepare the MODIS and AERONET data.')
        sys.exit(1)

    # ####################################################################
    # SETTINGS
    # ####################################################################
    num_gpus = 0  # set this if running on a machine that has GPUs
    num_workers = 0  # num_workers, sometimes 0 is the fastest
    Ntry = 20  # number of different random seeds in training

    with open('data/optimalnets.json', 'rt') as f:
        optimalnets = json.load(f)

    REGRlayers = optimalnets['REGR']
    CORRlayers = optimalnets['CORR']

    # ####################################################################

    # neural network model - 3 layers fully connected
    class DNN_3layers(pl.LightningModule):
        def __init__(self, Ninputs, N1sthidden, N2ndhidden, N3rdhidden, Noutputs, lr=5e-5):
            super().__init__()
            self.lr = lr
            self.nn = nn.Sequential(
                nn.Linear(Ninputs, N1sthidden),
                nn.ReLU(),
                nn.Linear(N1sthidden, N2ndhidden),
                nn.ReLU(),
                nn.Linear(N2ndhidden, N3rdhidden),
                nn.ReLU(),
                nn.Linear(N3rdhidden, Noutputs)
            )

        def forward(self, x):
            return self.nn(x)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

        def training_step(self, train_batch, batch_idx):
            x, y = train_batch
            x = x.view(x.size(0), -1)
            x_hat = self.nn(x)
            loss = F.mse_loss(x_hat, y)
            self.log('train_loss', loss, on_epoch=True, logger=True)
            return loss

        def validation_step(self, val_batch, batch_idx):
            x, y = val_batch
            x = x.view(x.size(0), -1)
            x_hat = self.nn(x)
            loss = F.mse_loss(x_hat, y)
            self.log('val_loss', loss, on_epoch=True, logger=True)
            return loss

    def dist(lat1, lon1, lat2, lon2):
        lat1R, lat2R, lon1R, lon2R = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon1), np.deg2rad(lon2)
        dlon = lon2R - lon1R
        dlat = lat2R - lat1R
        R = 6378.1
        a = (np.sin(dlat / 2.0))**2 + np.cos(lat1R) * np.cos(lat2R) * (np.sin(dlon / 2.0))**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        d = R * c
        return d

    # #########################################################################
    # Set graphics & other basic stuff
    # #########################################################################
    # some plotting parameters
    sns.set(style="darkgrid")
    sns.set_context("talk", font_scale=1.75, rc={"lines.linewidth": 2})
    params = {'axes.labelsize': 24,
              'axes.titlesize': 18,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20}
    plt.rcParams.update(params)
    # make dirs if does not exist
    os.makedirs('runfiles/figs', exist_ok=True)
    os.makedirs('runfiles/models', exist_ok=True)
    os.makedirs('runfiles/data', exist_ok=True)

    # #########################################################################
    # Prepare AERONET & MODIS data
    # #########################################################################
    if not os.path.isfile('runfiles/data/MODISdata_1000m_300s.h5'):
        print('Pre-processing AERONET-MODIS data...')
        # collocation settings for pixels to be saved in file (not the same as used in the actual processing)
        kmlimit = 1.0  # km
        timelimit = 300  # s = 5mins
        # Load data
        MODISdata = pd.read_hdf(os.path.join('data', 'DRAGON2011_MODISdf.h5'))
        AERONETdata = pd.read_hdf(os.path.join('data', 'DRAGON2011_AERONETdf.h5'))

        # compute MODIS-AERONET overpass distances & time difference
        MODISdata['timedelta'] = np.nan
        MODISdata['AERONETdistance'] = np.nan

        for ii, row in AERONETdata.iterrows():
            overpassID = row['overpassID']
            AERONET_lon, AERONET_lat = row['AERONET_lon'], row['AERONET_lat']
            overpassMASK = MODISdata['overpassID'] == overpassID
            thisoverpass_MODISdata = MODISdata[overpassMASK]
            MODISdata.loc[overpassMASK, ('AERONETdistance')] = dist(AERONET_lat, AERONET_lon, thisoverpass_MODISdata['MODIS_lat'], thisoverpass_MODISdata['MODIS_lon'])
            MODISdata.loc[overpassMASK, ('timedelta')] = row['AERONET_timestamp'] - thisoverpass_MODISdata['MODIS_Scan_Start_Time']

        MODISdataFiltered = MODISdata[(np.abs(MODISdata['timedelta']) < timelimit) & (MODISdata['AERONETdistance'] < kmlimit)]
        MODISdataFiltered.to_hdf('runfiles/data/MODISdata_1000m_300s.h5', key='df', mode='w')
        MODISdataFiltered = None
        print('Done!')

    # #########################################################################
    # HELPER FUNCTIONS
    # #########################################################################
    # Function to compute statistics for the results
    # #########################################################################
    def computeStats(true, predicted, computeEEratio=False):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')  # temporarily ignore this warning (would give warnings if predicted only contains a single value)
            N = len(true)
            R = np.corrcoef(true, predicted)[0, 1]
            R2 = R**2  # r2_score(true, predicted)
            RMSE = np.sqrt(np.mean((true - predicted)**2))
            BIAS = np.median(predicted - true)
            MAXABSERR = np.max(np.abs(predicted - true))
            EE = None
            if computeEEratio:
                EE = np.logical_and(predicted >= true * 0.85 - 0.05, predicted <= true * 1.15 + 0.05).sum() / N * 100.0
        return R, R2, RMSE, BIAS, MAXABSERR, EE

    # #########################################################################
    # Train neural network model Ntry times and save the best one
    # #########################################################################
    def trainModel(train_loader, val_loader, model, model_parameters, num_gpus, modelfilename):
        val_losses = []
        models = []
        for ii in range(Ntry):
            pl.seed_everything(ii)
            tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join('runfiles', 'lightning_logs'))
            models.append(model(**model_parameters))
            trainer = pl.Trainer(min_epochs=1,
                                 max_epochs=10000,
                                 gpus=num_gpus,
                                 enable_progress_bar=False,
                                 logger=tb_logger,
                                 callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=0, verbose=False, mode='min')])
            trainer.fit(models[-1], train_loader, val_loader)
            val_losses.append(trainer.validate(ckpt_path='best', dataloaders=val_loader)[0]['val_loss'])

        iiMinLoss = np.argmin(val_losses)
        torch.save(models[iiMinLoss].state_dict(), modelfilename)

    # #########################################################################
    # Load AERONET-collocated data
    # #########################################################################
    print('Loading data...')
    AERONETdata = pd.read_hdf(os.path.join('data', 'DRAGON2011_AERONETdf.h5'))
    data = pd.read_hdf(os.path.join('runfiles', 'data', 'MODISdata_1000m_300s.h5'))

    AERONETfields = ['AERONET_AOT_500', 'AERONET_440-870Angstrom', 'AERONET_station']
    for field in AERONETfields:
        data[field] = np.nan
    overpassIDs = list(set(data['overpassID']))
    for overpassID in overpassIDs:
        M = data['overpassID'] == overpassID
        AERONETrow = AERONETdata[AERONETdata['overpassID'] == overpassID].iloc[0]
        data.loc[M, 'AERONET_AOT_500'] = AERONETrow['AERONET_AOT_500']
        data.loc[M, 'AERONET_440-870Angstrom'] = AERONETrow['AERONET_440-870Angstrom']
        data.loc[M, 'AERONET_station'] = AERONETrow['AERONET_station']
    data['AERONET_AOT_550'] = data['AERONET_AOT_500'] * (550. / 500.)**-data['AERONET_440-870Angstrom']

    # accept only valid data & collocate (spatial collocation < 750m, temporal collocation < 250s, view zenith < 50 degrees)
    data = data[np.logical_and.reduce((~np.isnan(data['AERONET_AOT_550']), data['AERONETdistance'] <= 0.75, np.abs(data['timedelta']) < 250, data['MODIS_MOD04_3K_Sensor_Zenith'] < 50.0))]
    # accept 9 closest pixels for each MODIS-AERONET overpass
    data = data.groupby('overpassID').apply(lambda x: x.nsmallest(9, 'AERONETdistance')).reset_index(drop=True)
    print('ORIGINAL DATA LENGTH:', len(data))
    print('Number of AERONET sites: ', len(list(set(data['AERONET_station']))))
    print('Done!')

    # #########################################################################
    # Compute approximation errors
    # #########################################################################
    print('Computing approximation errors...')
    data['AOD550_approximationerror'] = data['AERONET_AOT_550'] - data['MODIS_MOD04_3K_Corrected_Optical_Depth_Land_1_smooth']  # AOD at 550 nm
    print('Done!')

    # #########################################################################
    # Split AERONET sites into two groups for cross-validation
    # #########################################################################
    print('Splitting AERONET data...')
    AERONETs = sorted(list(set(data['AERONET_station'])))
    AERONETlist1 = ['DRAGON_Aldino', 'DRAGON_Padonia', 'DRAGON_Essex', 'DRAGON_LAUMD', 'DRAGON_Edgewood', 'UMBC', 'DRAGON_ELLCT', 'DRAGON_Pylesville', 'DRAGON_CLRST', 'GSFC', 'DRAGON_BATMR', 'DRAGON_PineyOrchard']
    AERONETlist2 = ['SERC', 'DRAGON_ANNEA', 'DRAGON_UMRLB', 'DRAGON_EaglePoint', 'DRAGON_ARNCC', 'DRAGON_EDCMS', 'DRAGON_BLTCC', 'DRAGON_CLLGP', 'DRAGON_FairHill', 'Easton_Airport', 'DRAGON_BLDND', 'DRAGON_MNKTN']
    AERONETlist3 = ['DRAGON_Worton', 'DRAGON_WSTFD', 'DRAGON_BOWEM', 'DRAGON_PATUX', 'DRAGON_Beltsville', 'DRAGON_BTMDL', 'DRAGON_FLLST', 'DRAGON_SPBRK', 'DRAGON_KentIsland', 'DRAGON_ABERD', 'DRAGON_BLTNR', 'DRAGON_ARNLS', 'DRAGON_OLNES']
    # Originally the AERONET lists were selected randomly. To reproduce the results the lists are fixed here.
    # To use new random split of the AERONET sites, change useNewRandomSplit to True
    useNewRandomSplit = False
    if useNewRandomSplit:
        np.random.shuffle(AERONETs)
        AERONETlist1, AERONETlist2, AERONETlist3 = AERONETs[:int(len(AERONETs) / 3)], AERONETs[int(len(AERONETs) / 3):2 * int(len(AERONETs) / 3)], AERONETs[2 * int(len(AERONETs) / 3):]
    AERONETmask1, AERONETmask2, AERONETmask3 = data['AERONET_station'].isin(AERONETlist1), data['AERONET_station'].isin(AERONETlist2), data['AERONET_station'].isin(AERONETlist3)

    # #########################################################################
    # Variable name lists
    # #########################################################################
    GEOMETRYvariables = ['MODIS_MOD04_3K_Solar_Zenith', 'MODIS_MOD04_3K_Solar_Azimuth', 'MODIS_MOD04_3K_Sensor_Zenith', 'MODIS_MOD04_3K_Sensor_Azimuth', 'MODIS_MOD04_3K_Scattering_Angle', 'MODIS_MOD04_3K_Glint_Angle', 'MODIS_MOD04_3K_Topographic_Altitude_Land', 'MODIS_GMTED2010']
    SATELLITOBSERVATIONvariables = ['MODIS_TOAreflectance_band_01', 'MODIS_TOAreflectance_band_02', 'MODIS_TOAreflectance_band_03', 'MODIS_TOAreflectance_band_04', 'MODIS_TOAreflectance_band_05', 'MODIS_TOAreflectance_band_06', 'MODIS_TOAreflectance_band_07', 'MODIS_TOAreflectance_band_08', 'MODIS_TOAreflectance_band_09', 'MODIS_TOAreflectance_band_10', 'MODIS_TOAreflectance_band_11', 'MODIS_TOAreflectance_band_12', 'MODIS_TOAreflectance_band_13', 'MODIS_TOAreflectance_band_15', 'MODIS_TOAreflectance_band_19', 'MODIS_TOAreflectance_band_20', 'MODIS_TOAreflectance_band_21', 'MODIS_TOAreflectance_band_22']  # 'MODIS_TOAreflectance_band_14',  'MODIS_TOAreflectance_band_16', 'MODIS_TOAreflectance_band_17', 'MODIS_TOAreflectance_band_18', 
    LEVEL2variables = ['MODIS_MOD04_3K_Surface_Reflectance_Land_0', 'MODIS_MOD04_3K_Surface_Reflectance_Land_1', 'MODIS_MOD04_3K_Surface_Reflectance_Land_2', 'MODIS_MOD04_3K_Corrected_Optical_Depth_Land_0_smooth', 'MODIS_MOD04_3K_Corrected_Optical_Depth_Land_1_smooth', 'MODIS_MOD04_3K_Corrected_Optical_Depth_Land_2_smooth', 'MODIS_MOD04_3K_Corrected_Optical_Depth_Land_wav2p1_smooth']
    REGRoutputvariables = ['AERONET_AOT_550']
    CORRoutputvariables = ['AOD550_approximationerror']

    # #########################################################################
    # Prepare data for regression and correction models
    # #########################################################################
    print('Preparing data...')
    # (convert inputs & outputs to float to convert from "object" type (possible mixed variables: bools & floats) to numeric)
    REGRinputvariables = GEOMETRYvariables + SATELLITOBSERVATIONvariables
    inputsREGR1, outputsREGR1 = np.array(data[AERONETmask1][REGRinputvariables]).astype(float), np.array(data[AERONETmask1][REGRoutputvariables]).astype(float)
    inputsREGR2, outputsREGR2 = np.array(data[AERONETmask2][REGRinputvariables]).astype(float), np.array(data[AERONETmask2][REGRoutputvariables]).astype(float)
    inputsREGR3, outputsREGR3 = np.array(data[AERONETmask3][REGRinputvariables]).astype(float), np.array(data[AERONETmask3][REGRoutputvariables]).astype(float)
    CORRinputvariables = GEOMETRYvariables + SATELLITOBSERVATIONvariables + LEVEL2variables
    inputsCORR1, outputsCORR1 = np.array(data[AERONETmask1][CORRinputvariables]).astype(float), np.array(data[AERONETmask1][CORRoutputvariables]).astype(float)
    inputsCORR2, outputsCORR2 = np.array(data[AERONETmask2][CORRinputvariables]).astype(float), np.array(data[AERONETmask2][CORRoutputvariables]).astype(float)
    inputsCORR3, outputsCORR3 = np.array(data[AERONETmask3][CORRinputvariables]).astype(float), np.array(data[AERONETmask3][CORRoutputvariables]).astype(float)

    # #########################################################################
    # Test outputs for NaNs, there should not be any
    # #########################################################################
    if np.isnan(outputsCORR1).sum() > 0 or np.isnan(outputsCORR2).sum() > 0 or np.isnan(outputsCORR3).sum() > 0 or np.isnan(outputsREGR1).sum() > 0 or np.isnan(outputsREGR2).sum() > 0 or np.isnan(outputsREGR3).sum() > 0:
        raise Exception('NaN found in outputs')

    # #########################################################################
    # Fill in the missing input data (NaN)
    # #########################################################################
    # Use SimpleImputer for replacing NaNs with training data mean values
    imputerCORR1 = SimpleImputer(missing_values=np.nan, strategy='mean').fit(inputsCORR1)
    imputerCORR2 = SimpleImputer(missing_values=np.nan, strategy='mean').fit(inputsCORR2)
    imputerCORR3 = SimpleImputer(missing_values=np.nan, strategy='mean').fit(inputsCORR3)
    imputerREGR1 = SimpleImputer(missing_values=np.nan, strategy='mean').fit(inputsREGR1)
    imputerREGR2 = SimpleImputer(missing_values=np.nan, strategy='mean').fit(inputsREGR2)
    imputerREGR3 = SimpleImputer(missing_values=np.nan, strategy='mean').fit(inputsREGR3)

    # #########################################################################
    # Standard scalers to standardize inputs and outputs
    # #########################################################################
    scaler_inputs_REGR1 = StandardScaler().fit(inputsREGR1)
    scaler_inputs_REGR2 = StandardScaler().fit(inputsREGR2)
    scaler_inputs_REGR3 = StandardScaler().fit(inputsREGR3)
    scaler_outputs_REGR1 = StandardScaler().fit(outputsREGR1)
    scaler_outputs_REGR2 = StandardScaler().fit(outputsREGR2)
    scaler_outputs_REGR3 = StandardScaler().fit(outputsREGR3)
    scaler_inputs_CORR1 = StandardScaler().fit(inputsCORR1)
    scaler_inputs_CORR2 = StandardScaler().fit(inputsCORR2)
    scaler_inputs_CORR3 = StandardScaler().fit(inputsCORR3)
    scaler_outputs_CORR1 = StandardScaler().fit(outputsCORR1)
    scaler_outputs_CORR2 = StandardScaler().fit(outputsCORR2)
    scaler_outputs_CORR3 = StandardScaler().fit(outputsCORR3)

    print('Done!')

    # #########################################################################
    # Define the regression and correction models
    # #########################################################################
    print('Defining regression and correction models...')
    # ##################
    # REGRESSION MODELS
    # ##################

    # Transform to torch tensor
    tensor_inputsREGR1 = torch.Tensor(scaler_inputs_REGR1.transform(imputerREGR1.transform(inputsREGR1)))
    tensor_outputsREGR1 = torch.Tensor(np.reshape(scaler_outputs_REGR1.transform(outputsREGR1.ravel()[:, None]), outputsREGR1.shape))
    tensor_inputsREGR2 = torch.Tensor(scaler_inputs_REGR2.transform(imputerREGR2.transform(inputsREGR2)))
    tensor_outputsREGR2 = torch.Tensor(np.reshape(scaler_outputs_REGR2.transform(outputsREGR2.ravel()[:, None]), outputsREGR2.shape))
    tensor_inputsREGR3 = torch.Tensor(scaler_inputs_REGR3.transform(imputerREGR3.transform(inputsREGR3)))
    tensor_outputsREGR3 = torch.Tensor(np.reshape(scaler_outputs_REGR3.transform(outputsREGR3.ravel()[:, None]), outputsREGR3.shape))
    # Create datasets and loaders for them
    REGR1_dataset = TensorDataset(tensor_inputsREGR1, tensor_outputsREGR1)
    REGR1_loader = DataLoader(REGR1_dataset, batch_size=8, num_workers=num_workers)
    REGR2_dataset = TensorDataset(tensor_inputsREGR2, tensor_outputsREGR2)
    REGR2_loader = DataLoader(REGR2_dataset, batch_size=8, num_workers=num_workers)
    REGR3_dataset = TensorDataset(tensor_inputsREGR3, tensor_outputsREGR3)
    REGR3_loader = DataLoader(REGR3_dataset, batch_size=8, num_workers=num_workers)

    # ##################
    # CORRECTION MODELS
    # ##################

    # Transform to torch tensor
    tensor_inputsCORR1 = torch.Tensor(scaler_inputs_CORR1.transform(imputerCORR1.transform(inputsCORR1)))
    tensor_outputsCORR1 = torch.Tensor(np.reshape(scaler_outputs_CORR1.transform(outputsCORR1.ravel()[:, None]), outputsCORR1.shape))
    tensor_inputsCORR2 = torch.Tensor(scaler_inputs_CORR2.transform(imputerCORR2.transform(inputsCORR2)))
    tensor_outputsCORR2 = torch.Tensor(np.reshape(scaler_outputs_CORR2.transform(outputsCORR2.ravel()[:, None]), outputsCORR2.shape))
    tensor_inputsCORR3 = torch.Tensor(scaler_inputs_CORR3.transform(imputerCORR3.transform(inputsCORR3)))
    tensor_outputsCORR3 = torch.Tensor(np.reshape(scaler_outputs_CORR3.transform(outputsCORR3.ravel()[:, None]), outputsCORR3.shape))
    # Create datasets and loaders for them
    CORR1_dataset = TensorDataset(tensor_inputsCORR1, tensor_outputsCORR1)
    CORR1_loader = DataLoader(CORR1_dataset, batch_size=8, num_workers=num_workers)
    CORR2_dataset = TensorDataset(tensor_inputsCORR2, tensor_outputsCORR2)
    CORR2_loader = DataLoader(CORR2_dataset, batch_size=8, num_workers=num_workers)
    CORR3_dataset = TensorDataset(tensor_inputsCORR3, tensor_outputsCORR3)
    CORR3_loader = DataLoader(CORR3_dataset, batch_size=8, num_workers=num_workers)

    # save network layer sizes to a file
    with open(os.path.join('runfiles', 'networks.txt'), 'wt') as f:
        f.write('REGRlayers: {}\n'.format(REGRlayers))
        f.write('CORRlayers: {}\n'.format(CORRlayers))

    print('Done!')

    # #######################
    # Train regressor models!
    # #######################
    print('Training regressor models')
    model_parameters = {
        'Ninputs': inputsREGR1.shape[1],
        'N1sthidden': REGRlayers[0],
        'N2ndhidden': REGRlayers[1],
        'N3rdhidden': REGRlayers[2],
        'Noutputs': outputsREGR1.shape[1],
    }

    train_loader = REGR1_loader
    val_loader = REGR2_loader
    model = DNN_3layers
    modelfilename = os.path.join('runfiles', 'models', 'REGR_train1_val2_usefor3.pt')
    trainModel(train_loader, val_loader, model, model_parameters, num_gpus, modelfilename)

    train_loader = REGR2_loader
    val_loader = REGR3_loader
    model = DNN_3layers
    modelfilename = os.path.join('runfiles', 'models', 'REGR_train2_val3_usefor1.pt')
    trainModel(train_loader, val_loader, model, model_parameters, num_gpus, modelfilename)

    train_loader = REGR3_loader
    val_loader = REGR1_loader
    model = DNN_3layers
    modelfilename = os.path.join('runfiles', 'models', 'REGR_train3_val1_usefor2.pt')
    trainModel(train_loader, val_loader, model, model_parameters, num_gpus, modelfilename)

    print('Done!')

    # #######################
    # Train correction models!
    # #######################
    print('Training correction models')
    model_parameters = {
        'Ninputs': inputsCORR1.shape[1],
        'N1sthidden': CORRlayers[0],
        'N2ndhidden': CORRlayers[1],
        'N3rdhidden': CORRlayers[2],
        'Noutputs': outputsCORR1.shape[1],
    }

    train_loader = CORR1_loader
    val_loader = CORR2_loader
    model = DNN_3layers
    modelfilename = os.path.join('runfiles', 'models', 'CORR_train1_val2_usefor3.pt')
    trainModel(train_loader, val_loader, model, model_parameters, num_gpus, modelfilename)

    train_loader = CORR2_loader
    val_loader = CORR3_loader
    model = DNN_3layers
    modelfilename = os.path.join('runfiles', 'models', 'CORR_train2_val3_usefor1.pt')
    trainModel(train_loader, val_loader, model, model_parameters, num_gpus, modelfilename)

    train_loader = CORR3_loader
    val_loader = CORR1_loader
    model = DNN_3layers
    modelfilename = os.path.join('runfiles', 'models', 'CORR_train3_val1_usefor2.pt')
    trainModel(train_loader, val_loader, model, model_parameters, num_gpus, modelfilename)

    print('Done!')

    # #######################
    # Load trained models from files
    # #######################
    print('Loading trained models')

    modelREGR1 = DNN_3layers(Ninputs=inputsREGR1.shape[1], N1sthidden=REGRlayers[0], N2ndhidden=REGRlayers[1], N3rdhidden=REGRlayers[2], Noutputs=outputsREGR1.shape[1])
    modelREGR1.load_state_dict(torch.load(os.path.join('runfiles', 'models', 'REGR_train1_val2_usefor3.pt')))
    modelREGR1.eval()
    modelREGR2 = DNN_3layers(Ninputs=inputsREGR1.shape[1], N1sthidden=REGRlayers[0], N2ndhidden=REGRlayers[1], N3rdhidden=REGRlayers[2], Noutputs=outputsREGR1.shape[1])
    modelREGR2.load_state_dict(torch.load(os.path.join('runfiles', 'models', 'REGR_train2_val3_usefor1.pt')))
    modelREGR2.eval()
    modelREGR3 = DNN_3layers(Ninputs=inputsREGR1.shape[1], N1sthidden=REGRlayers[0], N2ndhidden=REGRlayers[1], N3rdhidden=REGRlayers[2], Noutputs=outputsREGR1.shape[1])
    modelREGR3.load_state_dict(torch.load(os.path.join('runfiles', 'models', 'REGR_train3_val1_usefor2.pt')))
    modelREGR3.eval()
    modelCORR1 = DNN_3layers(Ninputs=inputsCORR1.shape[1], N1sthidden=CORRlayers[0], N2ndhidden=CORRlayers[1], N3rdhidden=CORRlayers[2], Noutputs=outputsCORR1.shape[1])
    modelCORR1.load_state_dict(torch.load(os.path.join('runfiles', 'models', 'CORR_train1_val2_usefor3.pt')))
    modelCORR1.eval()
    modelCORR2 = DNN_3layers(Ninputs=inputsCORR1.shape[1], N1sthidden=CORRlayers[0], N2ndhidden=CORRlayers[1], N3rdhidden=CORRlayers[2], Noutputs=outputsCORR1.shape[1])
    modelCORR2.load_state_dict(torch.load(os.path.join('runfiles', 'models', 'CORR_train2_val3_usefor1.pt')))
    modelCORR2.eval()
    modelCORR3 = DNN_3layers(Ninputs=inputsCORR1.shape[1], N1sthidden=CORRlayers[0], N2ndhidden=CORRlayers[1], N3rdhidden=CORRlayers[2], Noutputs=outputsCORR1.shape[1])
    modelCORR3.load_state_dict(torch.load(os.path.join('runfiles', 'models', 'CORR_train3_val1_usefor2.pt')))
    modelCORR3.eval()

    print('Done!')

    # ###########################
    # Evaluate results
    # ###########################

    # Regression
    print('  Regression models...')
    predicted_aod_AERONET1 = scaler_outputs_REGR2.inverse_transform(modelREGR2.forward(torch.Tensor(scaler_inputs_REGR2.transform(imputerREGR2.transform(inputsREGR1)))).detach().numpy())
    predicted_aod_AERONET2 = scaler_outputs_REGR3.inverse_transform(modelREGR3.forward(torch.Tensor(scaler_inputs_REGR3.transform(imputerREGR3.transform(inputsREGR2)))).detach().numpy())
    predicted_aod_AERONET3 = scaler_outputs_REGR1.inverse_transform(modelREGR1.forward(torch.Tensor(scaler_inputs_REGR1.transform(imputerREGR1.transform(inputsREGR3)))).detach().numpy())
    AOD550_REGR = np.nan * np.ones(len(data))
    AOD550_REGR[AERONETmask1] = predicted_aod_AERONET1[:, 0]  # <- 0 = AOD 550 nm
    AOD550_REGR[AERONETmask2] = predicted_aod_AERONET2[:, 0]
    AOD550_REGR[AERONETmask3] = predicted_aod_AERONET3[:, 0]
    AOD550_REGR = np.clip(AOD550_REGR, a_min=0.005, a_max=np.inf)

    # Correction
    print('  Correction models...')
    predicted_approx_err_AERONET1 = scaler_outputs_CORR2.inverse_transform(modelCORR2.forward(torch.Tensor(scaler_inputs_CORR2.transform(imputerCORR2.transform(inputsCORR1)))).detach().numpy())
    predicted_approx_err_AERONET2 = scaler_outputs_CORR3.inverse_transform(modelCORR3.forward(torch.Tensor(scaler_inputs_CORR3.transform(imputerCORR3.transform(inputsCORR2)))).detach().numpy())
    predicted_approx_err_AERONET3 = scaler_outputs_CORR1.inverse_transform(modelCORR1.forward(torch.Tensor(scaler_inputs_CORR1.transform(imputerCORR1.transform(inputsCORR3)))).detach().numpy())
    AOD550_CORR = np.nan * np.ones(len(data))
    AOD550_CORR[AERONETmask1] = data['MODIS_MOD04_3K_Corrected_Optical_Depth_Land_1_smooth'][AERONETmask1] + predicted_approx_err_AERONET1[:, 0]  # <- 0 = AOD 550 nm
    AOD550_CORR[AERONETmask2] = data['MODIS_MOD04_3K_Corrected_Optical_Depth_Land_1_smooth'][AERONETmask2] + predicted_approx_err_AERONET2[:, 0]
    AOD550_CORR[AERONETmask3] = data['MODIS_MOD04_3K_Corrected_Optical_Depth_Land_1_smooth'][AERONETmask3] + predicted_approx_err_AERONET3[:, 0]
    AOD550_CORR = np.clip(AOD550_CORR, a_min=0.005, a_max=np.inf)

    print('M')
    print(M.sum())
    print(M.shape)
    print('***')
    M = np.logical_or.reduce((AERONETmask1, AERONETmask2, AERONETmask3))
    data = data[M]

    # Add results back to the original dataframe for plotting
    data['AOD550_REGR'] = AOD550_REGR
    data['AOD550_CORR'] = AOD550_CORR

    # #########################################################################
    # Compute statistics
    # #########################################################################
    N = len(data)
    AOD_R_DT, AOD_R2_DT, AOD_RMSE_DT, AOD_BIAS_DT, AOD_MAXABSERR_DT, AOD_EE_DT = computeStats(data['AERONET_AOT_550'], data['MODIS_MOD04_3K_Corrected_Optical_Depth_Land_1'], computeEEratio=True)
    AOD_R_REGR, AOD_R2_REGR, AOD_RMSE_REGR, AOD_BIAS_REGR, AOD_MAXABSERR_REGR, AOD_EE_REGR = computeStats(data['AERONET_AOT_550'], data['AOD550_REGR'], computeEEratio=True)
    AOD_R_CORR, AOD_R2_CORR, AOD_RMSE_CORR, AOD_BIAS_CORR, AOD_MAXABSERR_CORR, AOD_EE_CORR = computeStats(data['AERONET_AOT_550'], data['AOD550_CORR'], computeEEratio=True)
    print('Done!')

    # #########################################################################
    # Plot the figures
    # #########################################################################
    print('Plotting figures...')

    plotdata = [
        {
            'X': data['AERONET_AOT_550'],
            'Y': data['MODIS_MOD04_3K_Corrected_Optical_Depth_Land_1'],
            'xlabel': 'AOD, AERONET, 550 nm',
            'ylabel': 'AOD, DT, 550 nm',
            'ROW1': 'N = {} ({:.1f}% within DT EE envelope)'.format('{:,d}'.format(N).replace(',', ' '), AOD_EE_DT),
            'ROW2': 'R² = {:.3f}'.format(AOD_R2_DT),
            'ROW3': 'RMSE = {:.3f}'.format(AOD_RMSE_DT),
            'ROW4': 'BIAS = {:.3f}'.format(AOD_BIAS_DT),
            'ROW5': 'MAX(ABS(ERROR)) = {:.3f}'.format(AOD_MAXABSERR_DT),
            'xlim': [0.0, 0.8],
            'ylim': [-0.05, 0.8],
            'envelope': True
        },
        {
            'X': data['AERONET_AOT_550'],
            'Y': data['AOD550_REGR'],
            'xlabel': 'AOD, AERONET, 550 nm',
            'ylabel': 'AOD, Fully learned, 550 nm',
            'ROW1': 'N = {} ({:.1f}% within DT EE envelope)'.format('{:,d}'.format(N).replace(',', ' '), AOD_EE_REGR),
            'ROW2': 'R² = {:.3f}'.format(AOD_R2_REGR),
            'ROW3': 'RMSE = {:.3f}'.format(AOD_RMSE_REGR),
            'ROW4': 'BIAS = {:.3f}'.format(AOD_BIAS_REGR),
            'ROW5': 'MAX(ABS(ERROR)) = {:.3f}'.format(AOD_MAXABSERR_REGR),
            'xlim': [0.0, 0.8],
            'ylim': [-0.05, 0.8],
            'envelope': True
        },
        {
            'X': data['AERONET_AOT_550'],
            'Y': data['AOD550_CORR'],
            'xlabel': 'AOD, AERONET, 550 nm',
            'ylabel': 'AOD, post-process corrected DT, 550 nm',
            'ROW1': 'N = {} ({:.1f}% within DT EE envelope)'.format('{:,d}'.format(N).replace(',', ' '), AOD_EE_CORR),
            'ROW2': 'R² = {:.3f}'.format(AOD_R2_CORR),
            'ROW3': 'RMSE = {:.3f}'.format(AOD_RMSE_CORR),
            'ROW4': 'BIAS = {:.3f}'.format(AOD_BIAS_CORR),
            'ROW5': 'MAX(ABS(ERROR)) = {:.3f}'.format(AOD_MAXABSERR_CORR),
            'xlim': [0.0, 0.8],
            'ylim': [-0.05, 0.8],
            'envelope': True
        }
    ]

    fig = plt.figure(figsize=(24, 9.8), dpi=200)
    for ii in range(3):
        p = plotdata[ii]
        ax = fig.add_axes([0.045 + ii * 0.336, 0.1, 0.261, 0.87])
        ax.scatter(p['X'], p['Y'], marker='o', s=65, linewidth=2.0, facecolors='none', edgecolors='#8B0000', alpha=0.25, rasterized=True)
        ax.grid(True)
        ax.plot([-10, 10], [-10, 10], 'k-', linewidth=2.0, alpha=0.8, zorder=10)
        if 'envelope' in p and p['envelope']:
            ax.plot([-10, 10], [-10 * 1.15 + 0.05, 10 * 1.15 + 0.05], 'k--', linewidth=1.0, alpha=0.8, zorder=10)
            ax.plot([-10, 10], [-10 * 0.85 - 0.05, 10 * 0.85 - 0.05], 'k--', linewidth=1.0, alpha=0.8, zorder=10)
        ax.set_xlim(p['xlim'])
        ax.set_ylim(p['ylim'])
        p['ROW01'] = p['ROW1'].split(' (')[0]
        p['ROW02'] = p['ROW1'].split(' (')[1].replace(')', '')
        ax.text(0.99, 0.01 + 5 * 0.04, '{}'.format(p['ROW01']), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=22, color='k', zorder=99, bbox={'facecolor': 'white', 'alpha': 0.7, 'boxstyle': 'square,pad=0.03'})
        ax.text(0.99, 0.01 + 4 * 0.04, '{}'.format(p['ROW02']), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=22, color='k', zorder=99, bbox={'facecolor': 'white', 'alpha': 0.7, 'boxstyle': 'square,pad=0.03'})
        ax.text(0.99, 0.01 + 3 * 0.04, '{}'.format(p['ROW2']), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=22, color='k', zorder=99, bbox={'facecolor': 'white', 'alpha': 0.7, 'boxstyle': 'square,pad=0.03'})
        ax.text(0.99, 0.01 + 2 * 0.04, '{}'.format(p['ROW3']), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=22, color='k', zorder=99, bbox={'facecolor': 'white', 'alpha': 0.7, 'boxstyle': 'square,pad=0.03'})
        ax.text(0.99, 0.01 + 1 * 0.04, '{}'.format(p['ROW4']), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=22, color='k', zorder=99, bbox={'facecolor': 'white', 'alpha': 0.7, 'boxstyle': 'square,pad=0.03'})
        ax.text(0.99, 0.01 + 0 * 0.04, '{}'.format(p['ROW5']), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=22, color='k', zorder=99, bbox={'facecolor': 'white', 'alpha': 0.7, 'boxstyle': 'square,pad=0.03'})
        ax.set_xlabel(p['xlabel'])
        ax.set_ylabel(p['ylabel'])
    plt.savefig(os.path.join('runfiles', 'figs', 'AOD.pdf'))
    plt.close('all')

    print('Figures saved to directory "runfiles/figs".')
    print('Done!')
