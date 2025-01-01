from carbonai import PowerMeter
power_meter = PowerMeter(project_name="face recognition",
                         program_name="masque_with_resnet")

power_meter = PowerMeter.from_config(path="config.json")


@power_meter.measure_power(
    package="sklearn",
    algorithm="hello",
    data_type="tabular/images",
    data_shape="(1797, 64)",
    algorithm_params="loss='log', alpha=1e-5",
    comments="10 fold cross validated training of logistic regression classifier trained on the MNIST dataset"
)
def hello():
    print('hello')


hello()
