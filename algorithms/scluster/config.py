"""Config for the Spectral Clustering algorithm."""

# Spectral Clustering Params
config = {
    "num_layers" : 16, # 20 (n°1)
    "scluster_k" : 5,  
    "evec_smooth": 16, # 32 (n°1)
    "rec_smooth" : 16, # 16 (n°1)
    "rec_width"  : 1,
    #"evec_smooth": 16,
    #"rec_smooth" : 8,
    #"rec_width"  : 8,  

    "hier" : True
}

algo_id = "scluster"
is_boundary_type = True
is_label_type = True