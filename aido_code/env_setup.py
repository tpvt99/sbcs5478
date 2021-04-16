from gym_duckietown.simulator import Simulator

from aido_code.action_wrappers import ActionSmoothingWrapper, DiscreteWrapper, Heading2WheelVelsWrapper, LeftRightBraking2WheelVelsWrapper, \
    LeftRightClipped2WheelVelsWrapper, SteeringBraking2WheelVelsWrapper
from aido_code.observation_wrappers import NormalizeWrapper, ResizeWrapper, ClipImageWrapper
from aido_code.reward_wrappers import DtRewardClipperWrapper, DtRewardCollisionAvoidance, DtRewardPosAngle, DtRewardProximityPenalty, \
    DtRewardTargetOrientation, DtRewardVelocity, DtRewardWrapperDistanceTravelled

def setup(configs):

    env = Simulator(
            seed=configs['seed'],  # random seed
            map_name=configs['map'],
            max_steps=configs['ep_len'],
            domain_rand=False
            )

    # Observation Wrappers
    #if configs['env_config']['crop_image_top']:
    #    env = ClipImageWrapper(env)
    env = ResizeWrapper(env, configs['env_config']['resized_image'])
    #env = NormalizeWrapper(env)

    # Action wrappers
    if configs['env_config']["action_type"] == 'discrete':
        env = DiscreteWrapper(env)
    elif 'heading' in configs['env_config']["action_type"]:
        env = Heading2WheelVelsWrapper(env, configs['env_config']["action_type"])
    elif configs['env_config']["action_type"] == 'leftright_braking':
        env = LeftRightBraking2WheelVelsWrapper(env)
    elif configs['env_config']["action_type"] == 'leftright_clipped':
        env = LeftRightClipped2WheelVelsWrapper(env)
    elif configs['env_config']["action_type"] == 'steering_braking':
        env = SteeringBraking2WheelVelsWrapper(env)

    # Reward wrappers
    if configs['env_config']["reward_function"] in ['Posangle', 'posangle']:
        env = DtRewardPosAngle(env)
        env = DtRewardVelocity(env)
    elif configs['env_config']["reward_function"] == 'target_orientation':
        env = DtRewardTargetOrientation(env)
        env = DtRewardVelocity(env)
    elif configs['env_config']["reward_function"] == 'lane_distance':
        env = DtRewardWrapperDistanceTravelled(env)
    elif configs['env_config']["reward_function"] == 'default_clipped':
        env = DtRewardClipperWrapper(env, 2, -2)
    env = DtRewardCollisionAvoidance(env)

    # env = DtRewardProximityPenalty(env)
    return env

def setup_for_test(configs):

    env = Simulator(
            seed=2,#configs['seed'],  # random seed
            map_name=configs['map'],
            max_steps=configs['ep_len'],
            domain_rand=False)

    # Observation Wrappers
    #if configs['env_config']['crop_image_top']:
    #    env = ClipImageWrapper(env)
    env = ResizeWrapper(env, (64,64))
    #env = NormalizeWrapper(env)

    # Action wrappers
    # Action wrappers
    if configs['env_config']["action_type"] == 'discrete':
        env = DiscreteWrapper(env)
    elif 'heading' in configs['env_config']["action_type"]:
        env = Heading2WheelVelsWrapper(env, configs['env_config']["action_type"])
    elif configs['env_config']["action_type"] == 'leftright_braking':
        env = LeftRightBraking2WheelVelsWrapper(env)
    elif configs['env_config']["action_type"] == 'leftright_clipped':
        env = LeftRightClipped2WheelVelsWrapper(env)
    elif configs['env_config']["action_type"] == 'steering_braking':
        env = SteeringBraking2WheelVelsWrapper(env)

    if configs['env_config']["reward_function"] in ['Posangle', 'posangle']:
        env = DtRewardPosAngle(env)
        env = DtRewardVelocity(env)
    elif configs['env_config']["reward_function"] == 'target_orientation':
        env = DtRewardTargetOrientation(env)
        env = DtRewardVelocity(env)
    elif configs['env_config']["reward_function"] == 'lane_distance':
        env = DtRewardWrapperDistanceTravelled(env)
    elif configs['env_config']["reward_function"] == 'default_clipped':
        env = DtRewardClipperWrapper(env, 2, -2)
    env = DtRewardCollisionAvoidance(env)


    return env