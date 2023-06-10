from pathlib import Path
class represent_param:
    name = "representation"
    x = [
        "实际功率(MW)",
        "10米高度处风速（m/s）",
        "10米高度处风向（°）",
        "30米高度处风速（m/s）",
        "30米高度处风向（°）",
        "50米高度处风速（m/s）",
        "50米高度处风向（°）",
        "70米高度处风速（m/s）",
        "70米高度处风向（°）",
        "风机轮毂高度处风速（m/s）",
        "风机轮毂高度处风向（°）",
        "气温（°C）",
        "气压（hpa）",
        "相对湿度（%）",
    ]
    target = "实际功率(MW)"
    data_p = "" # the path of train data

    model_p = "" # the model of pre-training stage
    batch_size = 64 
    learning_rate = 2e-5
    pred_len = 6
    label_len = 0
    standard = True

class reg_param:
    name = "regression"
    x = [
        "实际功率(MW)",
        "10米高度处风速（m/s）",
        "10米高度处风向（°）",
        "30米高度处风速（m/s）",
        "30米高度处风向（°）",
        "50米高度处风速（m/s）",
        "50米高度处风向（°）",
        "70米高度处风速（m/s）",
        "70米高度处风向（°）",
        "风机轮毂高度处风速（m/s）",
        "风机轮毂高度处风向（°）",
        "气温（°C）",
        "气压（hpa）",
        "相对湿度（%）",
    ]
    target = "实际功率(MW)"
    data_p = "" # the path of train data
    model = "" # the model of the regression stage
    model_represent = "" # the well-trained model of the pre-training stage
    batch_size = 64 
    learning_rate = 2e-5
    pred_len = 6
    label_len = 0
    standard = True
