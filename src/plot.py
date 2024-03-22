from deep_dating.summary import *


if __name__ == "__main__":
    # p = read_serialize("runs_v2/graphs/pipeline_results.pkl")
    # pipeline_compare(*p)
    
    #binet_loss("runs_v2/Binet_aug_norm/epoch_log_split_0_Mar5-11-56-12.csv")
    
    binet_compare()

    #binet_synthetic()

    #feature_vis = FeatureVis("runs_v2/MPS_P1_Crossval/model_epoch_2_split_3_feats_val_split_3.pkl")
    #feature_vis.tsne()