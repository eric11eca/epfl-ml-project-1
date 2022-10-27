import os
from pickle import NONE
import matplotlib.pyplot as plt 
import seaborn as sns

def plot_training_stats(kfold_training_tracker, save_path, title=None):
    plt.figure(figsize=(16, 8))
    # print("--------------------------")
    # print("| \t BEST Result \t | ")
    # print("--------------------------")
    res_names_set_dict = {
                'Loss': ['train_loss', 'val_loss'],
                'Accuracy': ['train_acc', 'val_acc'],
                'Prec/Recall': ['val_precision', 'val_recall', 'val_f1']
            }
    for i, (key, res_names) in enumerate(res_names_set_dict.items()):
        legends = []
        plt.subplot(1,3,i+1)
        for name in res_names:
            for kfold in kfold_training_tracker.keys():
                stats_record = kfold_training_tracker[kfold][name]
                plt.plot(stats_record)
                legends.append(f'{name}-{kfold}')
        plt.legend(legends)
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.title(key)
    if title is not None:
        plt.suptitle(title)
    plt.savefig(save_path)
    print(f'Figure Saved => {save_path} ;)')



def plot_weights(feature_weights_dict:dict, save_path:str):
    plt.figure(figsize=(16, 8))
    poly_factor = 4
    # importances = [-1.32965182458636, 1.59305484472679, -1.74082999592372, -1.43615958824473, 1.52745888951044, -0.353520747184024, -1.42191602061363, 2.65743411760791, -1.32859639274237, 3.01146839745888, -6.08731144369581, -0.0873109623509647, 2.47838561272792, 0.200720258965311, 0.447477618556467, -0.610357046024988, 0.278396669617005, -0.836989450905287, 0.811566906942274, -0.110576959813089, -0.109159946263693, -0.615780278349467, 0.299154780759724, 0.912701819947148, -0.708280576790017, -0.341019986067839, -0.133358819527409, 0.131870184027089, 0.111657060498182, 3.65967172252464, -1.62136131224347, -3.18085422858401, 2.33115419153334, -0.0924255350556533, -0.306825668288548, 1.01383521299578, -0.542233218647816, 0.434439215570512, -0.113959872872515, -0.148809833101012, -0.0214334410087955, -1.33097385044766, 1.13008995598636, -0.604168619092488, 0.152391799981412, 0.11627416430794, 0.0877335833143279, 0.0476461512591046, 0.0226290735490327, -0.265519051970197, 0.642584955597092, 0.255975284996528, -0.423557403325891, 0.509639635410738, -0.0124440600688991, -0.52646202117385, 0.43358864873163, -0.00800578891877375, -0.0701889884981175, -0.0140174476199304, -0.036293002935025, -0.0429147777335379, -0.00524784667456433, -0.00400722491569976, 0.0328729029676728, 0.953348460561524, -0.15286208114669, -0.110287811237973, 0.155122584601506, 0.0302985405279172, -0.205827302708149, -0.0403233243317214, -0.0337758458266575, -5.27142646897729e-05, -0.00942454754412016, -0.0505595707867396, 0.0225587171300265, -0.32459753422942, 2.00379164922799, -1.98631747221384, 0.71897559237231, -0.0054553504428437, 0.0431059061325494, -0.0038884625378598, -0.0398505095212108, 0.147978774693074, -0.306747678068986, -0.347712673216657, 0.403268186515302, 1.14303428641372, -0.925044659772962, 0.056066202084089, 0.285967049683176, -0.0282698146405209, 0.559744138998769, 0.0480557769575305, -0.185561327357994, -0.0672756678466239, -0.00894830983780918, 0.0013439834378533, -0.0438480059930942, 0.374376535916473, 0.112928830623288, -0.333331161796267, 0.178277741284389, -0.00709399602559548, 0.544052003776468, 0.0469612964315505, -0.182969960678796, 0.00235312602927989, -0.0583923931734629, 0.00630800607696066, 0.0360778758919943, -0.312949195277552, -0.57750469868881, 0.8353814729458, -0.297588911922569, -0.924564297082086, -0.583708903268174, -0.613150580210792, 0.751313270360318]
    importances = feature_weights_dict.values()
    feature_names = feature_weights_dict.keys()
    # feature_names = ['x_bias', 'DER_mass_MMC_1', 'DER_mass_MMC_2', 'DER_mass_MMC_3', 'DER_mass_MMC_4', 'DER_mass_transverse_met_lep_1', 'DER_mass_transverse_met_lep_2', 'DER_mass_transverse_met_lep_3', 'DER_mass_transverse_met_lep_4', 'DER_mass_vis_1', 'DER_mass_vis_2', 'DER_mass_vis_3', 'DER_mass_vis_4', 'DER_pt_h_1', 'DER_pt_h_2', 'DER_pt_h_3', 'DER_pt_h_4', 'DER_deltaeta_jet_jet_1', 'DER_deltaeta_jet_jet_2', 'DER_deltaeta_jet_jet_3', 'DER_deltaeta_jet_jet_4', 'DER_mass_jet_jet_1', 'DER_mass_jet_jet_2', 'DER_mass_jet_jet_3', 'DER_mass_jet_jet_4', 'DER_prodeta_jet_jet_1', 'DER_prodeta_jet_jet_2', 'DER_prodeta_jet_jet_3', 'DER_prodeta_jet_jet_4', 'DER_deltar_tau_lep_1', 'DER_deltar_tau_lep_2', 'DER_deltar_tau_lep_3', 'DER_deltar_tau_lep_4', 'DER_pt_tot_1', 'DER_pt_tot_2', 'DER_pt_tot_3', 'DER_pt_tot_4', 'DER_sum_pt_1', 'DER_sum_pt_2', 'DER_sum_pt_3', 'DER_sum_pt_4', 'DER_pt_ratio_lep_tau_1', 'DER_pt_ratio_lep_tau_2', 'DER_pt_ratio_lep_tau_3', 'DER_pt_ratio_lep_tau_4', 'DER_met_phi_centrality_1', 'DER_met_phi_centrality_2', 'DER_met_phi_centrality_3', 'DER_met_phi_centrality_4', 'DER_lep_eta_centrality_1', 'DER_lep_eta_centrality_2', 'DER_lep_eta_centrality_3', 'DER_lep_eta_centrality_4', 'PRI_tau_pt_1', 'PRI_tau_pt_2', 'PRI_tau_pt_3', 'PRI_tau_pt_4', 'PRI_tau_eta_1', 'PRI_tau_eta_2', 'PRI_tau_eta_3', 'PRI_tau_eta_4', 'PRI_tau_phi_1', 'PRI_tau_phi_2', 'PRI_tau_phi_3', 'PRI_tau_phi_4', 'PRI_lep_pt_1', 'PRI_lep_pt_2', 'PRI_lep_pt_3', 'PRI_lep_pt_4', 'PRI_lep_eta_1', 'PRI_lep_eta_2', 'PRI_lep_eta_3', 'PRI_lep_eta_4', 'PRI_lep_phi_1', 'PRI_lep_phi_2', 'PRI_lep_phi_3', 'PRI_lep_phi_4', 'PRI_met_1', 'PRI_met_2', 'PRI_met_3', 'PRI_met_4', 'PRI_met_phi_1', 'PRI_met_phi_2', 'PRI_met_phi_3', 'PRI_met_phi_4', 'PRI_met_sumet_1', 'PRI_met_sumet_2', 'PRI_met_sumet_3', 'PRI_met_sumet_4', 'PRI_jet_leading_pt_1', 'PRI_jet_leading_pt_2', 'PRI_jet_leading_pt_3', 'PRI_jet_leading_pt_4', 'PRI_jet_leading_eta_1', 'PRI_jet_leading_eta_2', 'PRI_jet_leading_eta_3', 'PRI_jet_leading_eta_4', 'PRI_jet_leading_phi_1', 'PRI_jet_leading_phi_2', 'PRI_jet_leading_phi_3', 'PRI_jet_leading_phi_4', 'PRI_jet_subleading_pt_1', 'PRI_jet_subleading_pt_2', 'PRI_jet_subleading_pt_3', 'PRI_jet_subleading_pt_4', 'PRI_jet_subleading_eta_1', 'PRI_jet_subleading_eta_2', 'PRI_jet_subleading_eta_3', 'PRI_jet_subleading_eta_4', 'PRI_jet_subleading_phi_1', 'PRI_jet_subleading_phi_2', 'PRI_jet_subleading_phi_3', 'PRI_jet_subleading_phi_4', 'PRI_jet_all_pt_1', 'PRI_jet_all_pt_2', 'PRI_jet_all_pt_3', 'PRI_jet_all_pt_4', 'jetnum3', 'jetnum2', 'jetnum1', 'jetnum0']
    assert len(importances) == len(feature_names)
    num_poly_features = len(feature_names) - 1 - 4
    float_color_names = ['C' + str(i) for i in range(1, num_poly_features + 1)]
    poly_float_color_names = [item for item in float_color_names for i in range(poly_factor)]
    colors = ['C0'] + poly_float_color_names + ['C' + str(num_poly_features + 1)]
    fig, ax = plt.subplots()
    plt.xticks(rotation='vertical')
    plt.bar(feature_names, importances, color=colors)
    fig.set_size_inches(30, 6)
    ax.set_ylabel("Importance")
    fig.tight_layout()
    fig.savefig(save_path)
