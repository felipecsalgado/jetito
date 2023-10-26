import os
from jetito.ebeam import emittance

basepath = os.path.abspath(os.path.dirname(__file__)) + "/"

# G4 Selection 1 (all particles)
file = basepath + "images/Emittance_Experiment_2022/G4/" + \
    "G4sims+lwfa_PP_calib=18.5um_px.png"

pp_anaylsis = emittance.pp_emittance_calculator(file, image_calib=18.5e-3)

pp_anaylsis.crop_image(left=0, right=-1, top=0, bottom=-1, axis='xaxis',
                       save_file=basepath +
                       "results/ebeam/emittance-pp/ebeam_emittance_G4_cropped.png",
                       verbose=True)

pp_anaylsis.process(baseline_deg=10,
                    distance=25,
                    height=250,
                    del_peaks=None)

pp_anaylsis.compute(verbose=True)

# G4 Selection 2 (divergence < 10 mrad)
file = basepath + "images/Emittance_Experiment_2022/G4/" + \
    "G4sims+lwfa_PP_calib=18.5um_px_sel2.png"

pp_anaylsis.plot_analysis(save_file=basepath +
                          "results/ebeam/emittance-pp/ebeam_emittance_G4_analysis_sel2.png")

pp_anaylsis = emittance.pp_emittance_calculator(file, image_calib=18.5e-3)

pp_anaylsis.crop_image(left=0, right=-1, top=0, bottom=-1, axis='xaxis',
                       save_file=basepath +
                       "results/ebeam/emittance-pp/ebeam_emittance_G4_cropped_sel2.png",
                       verbose=True)

pp_anaylsis.process(baseline_deg=10,
                    distance=25,
                    height=250,
                    del_peaks=None)

pp_anaylsis.compute(verbose=True)

pp_anaylsis.plot_analysis(save_file=basepath +
                          "results/ebeam/emittance-pp/ebeam_emittance_G4_analysis_sel2.png")

# PP Experiment data
file = basepath + "images/Emittance_Experiment_2022/pp/" + \
    "shot00750.tif"

pp_anaylsis = emittance.pp_emittance_calculator(file, image_calib=18.5e-3)

pp_anaylsis.crop_image(left=0, right=1000, top=850, bottom=1965, axis='xaxis',
                       save_file=basepath +
                       "results/ebeam/emittance-pp/ebeam_emittance_PP_cropped.png",
                       verbose=True)

pp_anaylsis.process(baseline_deg=11,
                    savgol_deg=-1,
                    disk_r=3,
                    distance=40,
                    height=300,
                    del_peaks=(16, 17))

pp_anaylsis.compute(verbose=True)
pp_anaylsis.plot_analysis(save_file=basepath +
                          "results/ebeam/emittance-pp/ebeam_emittance_PP_analysis.png")
