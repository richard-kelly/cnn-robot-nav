
define hokuyolaser ranger
(
    # laser-specific properties
    sensor( 			
        range [ 0.10  30.0 ]
        fov 270.0
        samples 1080
    )

    # generic model properties

    # Draw a block to represent the physical sensor
    block( 
        points 4 
        point[0] [ -0.05 -0.05 ]
        point[1] [ 0.05 -0.05 ]
        point[2] [ 0.05 0.05 ]
        point[3] [ -0.05 0.05 ]
        z [-0.05 0.02]
    )

    color "blue"
    size [ 0.05 0.05 0.07 ]
)
