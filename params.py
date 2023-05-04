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
    data_p = Path("~/bbbb/data")

    model_p = "~/bbbb/model/represent_lstm"
    batch_size = 64  # 88888: 3, m_loss
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
    data_p = Path("~/bbbb/data")
    model = "/bbbb/model/regression_lstm"
    model_represent = "/bbbb/model/represent_lstm"
    batch_size = 64  # 88888: 3, m_loss
    learning_rate = 2e-5
    pred_len = 6
    label_len = 0
    standard = True
