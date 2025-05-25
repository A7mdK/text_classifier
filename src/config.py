MODELS = {
     "huawei-noah/TinyBERT_General_4L_312D",  
    # "distilbert-base-uncased",
    # "distilroberta-base",  
    # "google/electra-small-discriminator",  
    # "albert-tiny"

    # "roberta-base"  
    # "GroNLP/hateBERT"
}

DATASETS = {
    ### Human Written or AI Generated
    # "HC3": ("Hello-SimpleAI/HC3", "all"),
    # "AI-and-Human-Generated-Text": "Ateeqq/AI-and-Human-Generated-Text",
    # "ai-human-gen": "likhithasapu/ai-human-gen",
    #----------------------------------------------------------------------
    ### Hate Speech and Offensive Language Detection
    "davidson_offensive": "tdavidson/hate_speech_offensive",
    # "gender_hate": "ctoraman/gender-hate-speech",
    # "bad_hate": "badmatr11x/hate-offensive-speech"
}

RESULT_CONFIG = {
    "save_dir": "./results",
    "formats": ["plot"],
    "metrics": ["accuracy", "f1"] 
}

TRAINING_PRESETS = {
    "default": {
        "batch_size": 16,
        "epochs": 1,
        "max_length": 128,
        "learning_rate": 2e-5,
        "weight_decay": 0.01
    },
    "batch_32": {
        "batch_size": 32,
        "epochs": 1,
        "max_length": 128,
        "learning_rate": 2e-5,
        "weight_decay": 0.01
    },
    "max_lenght_256": {
        "batch_size": 16,
        "epochs": 1,
        "max_length": 256,
        "learning_rate": 2e-5,
        "weight_decay": 0.01
    },
    "epochs_3": {
        "batch_size": 16,
        "epochs": 3,
        "max_length": 128,
        "learning_rate": 2e-5,
        "weight_decay": 0.01
    },
    "learning_rate_3e-5": {
        "batch_size": 16,
        "epochs": 1,
        "max_length": 128,
        "learning_rate": 3e-5,
        "weight_decay": 0.01
    },
    "learning_rate_5e-5": {
        "batch_size": 16,
        "epochs": 1,
        "max_length": 128,
        "learning_rate": 5e-5,
        "weight_decay": 0.01
    },
    "weight_decay_0.001": {
        "batch_size": 16,
        "epochs": 1,
        "max_length": 128,
        "learning_rate": 2e-5,
        "weight_decay": 0.001
    },
    "weight_decay_0.1": {
        "batch_size": 16,
        "epochs": 1,
        "max_length": 128,
        "learning_rate": 2e-5,
        "weight_decay": 0.1
    },
    "high_performance": {
        "batch_size": 32,
        "epochs": 3,
        "max_length": 256,
        "learning_rate": 3e-5,
        "weight_decay": 0.01
    },
    "quick_test": {
        "batch_size": 8,
        "epochs": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "max_length": 64
    },
    "special": {
        "batch_size": 32,
        "epochs": 3,
        "max_length": 512,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        #"metric_for_best_model": "eval_f1",
        #"lr_scheduler_type": "linear",  # or "cosine"
        #"warmup_ratio": 0.1,
        # "early_stopping_patience": 1,
        # "metric": "f1_macro"
    },
}