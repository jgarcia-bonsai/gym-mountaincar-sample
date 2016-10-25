schema GameState
    Float32 x_position,
    Float32 x_velocity
end

schema Action
    Int8{0, 1, 2} command
end

schema MountainCarConfig
    Int8 episode_length,
    Int8 num_episodes,
    UInt8 deque_size
end

simulator mountaincar_simulator(MountainCarConfig)
    action (Action)
    state (GameState)
end

concept high_score is classifier
    predicts (Action)
    follows input(GameState)
    feeds output
end

curriculum high_score_curriculum
    train high_score
    with simulator mountaincar_simulator
    objective open_ai_gym_default_objective

        lesson get_high_score
            configure
                constrain episode_length with Int8{-1},
                constrain num_episodes with Int8{-1},
                constrain deque_size with UInt8{1}
            until
                maximize open_ai_gym_default_objective
end
