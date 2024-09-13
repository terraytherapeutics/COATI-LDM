FIGURE_DATA_PATH = "s3://terray-public/coatiLDM/figure_data"
DIFFUSION_MODELS = {
    "diffusion_large": "s3://terray-public/coatiLDM/models/general_coati_diffuser.pt",
    "hcaii_diff": "s3://terray-public/coatiLDM/models/hcaii_diff.pt",
    "logp_diff": "s3://terray-public/coatiLDM/models/logp_diff.pt",
    "tpsa_diff": "s3://terray-public/coatiLDM/models/tpsa_diff.pt",
    "logp_tpsa_diff": "s3://terray-public/coatiLDM/models/logp_tpsa_diff.pt",
    "uncond_diff": "s3://terray-public/coatiLDM/models/uncond_diff.pt",
    "qed_diff": "s3://terray-public/coatiLDM/models/qed_opt_diffuser.pt",
}

FLOW_MODELS = {
    "flow_large": "s3://terray-public/coatiLDM/models/general_coati_flow.pt",
    "logp_flow": "s3://terray-public/coatiLDM/models/logp_flow.pt",
    "tpsa_flow": "s3://terray-public/coatiLDM/models/tpsa_flow.pt",
    "hcaii_flow": "s3://terray-public/coatiLDM/models/hcaii_flow.pt",
    "logp_tpsa_flow": "s3://terray-public/coatiLDM/models/logp_tpsa_flow.pt",
    "uncond_flow": "s3://terray-public/coatiLDM/models/uncond_flow.pt",
    "uncond_for_dflow": "s3://terray-public/coatiLDM/models/uncond_for_dflow.pt",
}

COATI2_DOCS = {
    "general_doc": "s3://terray-public/coatiLDM/models/general_doc.pt",
    "qed_doc": "s3://terray-public/coatiLDM/models/qed_doc.pt",
}

CLASSIFIER_GUIDE_DOCS = {
    "tpsa": "s3://terray-public/coatiLDM/models/tpsa_cg.pt",
    "logp": "s3://terray-public/coatiLDM/models/logp_cg.pt",
    "hcaii": "s3://terray-public/coatiLDM/models/hcaii_cg.pt",
    "qed": "s3://terray-public/coatiLDM/models/guide.pt",
}

DFLOW_CLASSIFIER_DOCS = {
    "hcaii": "s3://terray-public/coatiLDM/models/dflow_hcaii_due.pt",
    "logp": "s3://terray-public/coatiLDM/models/dflow_logp_due.pt",
    "tpsa": "s3://terray-public/coatiLDM/models/dflow_tpsa_due.pt",
}

QED_OPT_DOCS = {
    "score_model": "s3://terray-public/coatiLDM/models/qed_opt_diffuser.pt",
    "guide": "s3://terray-public/coatiLDM/models/guide.pt",
}
