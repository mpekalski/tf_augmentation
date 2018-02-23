def augmentation( img, labels, target_size, do_nothing_prob = 0.5, transformations = [], num_transformations=3
                 , available_transofrmations=['distort','translate','crop','h_flip', 'v_flip', 'transpose'
                                              ,'rotate','normalize', 'brightness', 'contrast']
                 , TOTAL_TRANSFORMATIONS=10, TRANSLATE_BY_X=25, TRANSLATE_BY_Y=25):  
        with tf.device('/device:CPU:0'):
            # take randomly some transformations to apply    
            # order will also be random
            #if not transformations:
            #    y = tf.cond(tf.greater(tf.random_uniform(minval=0.0,maxval=1.0,dtype=tf.float32,shape=[]),  do_nothing_prob),
            #        y = np.random.choice(range(TOTAL_TRANSFORMATIONS), np.random.choice(range(num_transformations)), replace=False)
            #        transformations = [available_transofrmations[yy] for yy in y]
            #        with open(str(y),'w') as f:
            #            f.write(str(transformations))

            distorted_image = img
            # Image processing for training the network. Note the many random
            # distortions applied to the image.
            #10tf.image.flip_up_down
            #tf.image.random_flip_up_down
            #tf.image.flip_left_right
            #tf.image.random_flip_left_right
            #tf.image.transpose_image
            #tf.image.rot90
            #tf.image.resize_image_with_crop_or_pad
            #tf.image.central_crop
            #tf.image.pad_to_bounding_box
            #tf.image.crop_to_bounding_box
            #tf.image.extract_glimpse
            #tf.image.crop_and_resize
            #distorted_image =tf.contrib.image.angles_to_projective_transforms(img, 160,160)
            # Because most of thoe transformations are not commutative we randomize the order
            def rotate():
                    rnd_angle = tf.random_uniform(
                        [1],
                        minval=-20* np.pi / target_size,
                        maxval=20* np.pi / target_size,
                        dtype=tf.float32,
                        name='random_angle'
                    )

                    return tf.contrib.image.rotate(
                            distorted_image,
                            rnd_angle,
                            interpolation='NEAREST'
                            )
            def crop():
                return tf.random_crop(distorted_image, [target_size, target_size, 3])

            def h_flip():
                return tf.image.flip_left_right(distorted_image)                

            def v_flip():
                return tf.image.flip_up_down(distorted_image)

            def transpose():
                return tf.image.transpose_image(distorted_image)

            def brightness():
                return tf.image.random_brightness(distorted_image, max_delta=63)

            def contrast():
                return tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

            def normalize():
            # Subtract off the mean and divide by the variance of the pixels.
                return tf.image.per_image_standardization(distorted_image)

            def distort():
                return tf.image.central_crop(distorted_image, 0.6)

            def negative():
                return tf.subtract(255., distorted_image)

            def identity():
                return distorted_image

            def random_shift_h():
                width_shift_range = 0.3
                tx = tf.multiply(tf.random_uniform( shape=[], minval=-width_shift_range, maxval=width_shift_range), target_size)
                return tf.contrib.image.transform(distorted_image, transforms=[1, 0, tx, 0, 1, 0, 0, 0])
            
            def random_shift_v():
                width_shift_range = 0.3
                ty = tf.multiply(tf.random_uniform( shape=[], minval=-width_shift_range, maxval=width_shift_range), target_size)                    
                return tf.contrib.image.transform(distorted_image, transforms=[1, 0, 0, 0, 1, ty, 0, 0])           
            
            # Uniform variable in [0,1)

            threshold = tf.constant(0.9, dtype=tf.float32)
            
            def body(i, distorted_image):
                p_order = tf.random_uniform(shape=[6], minval=0., maxval=1., dtype=tf.float32)
                distorted_image = tf.case({
                                        
                                       tf.greater(p_order[0], threshold): rotate, 
                                       tf.greater(p_order[1], threshold): h_flip, 
                                       tf.greater(p_order[2], threshold): random_shift_v,
                                       tf.greater(p_order[3], threshold): random_shift_h,
                                       tf.greater(p_order[4], threshold): transpose, 
                                       tf.greater(p_order[5], threshold): v_flip, 
                                       #tf.greater(p_order[2], threshold): translate, 
                                       #tf.greater(p_order[3], threshold): contrast, 
                                       #tf.greater(p_order[4], threshold): distort, 
                                       #tf.greater(p_order[6], threshold): brightness, 
                                       #tf.greater(p_order[7], threshold): normalize, 
                                       #tf.greater(p_order[8], threshold): negative, 
                                        }
                                     ,default=identity, exclusive=False)
                return (i+1, distorted_image)

            def cond(i, *args):
                return i < 4 # num_transformations
            parallel_iterations = 1
            tf.while_loop(cond, body, [0,distorted_image], parallel_iterations=parallel_iterations)


            # expanding dims might be needed for keras
            #float_image = tf.expand_dims(tf.image.convert_image_dtype(distorted_image, dtype=tf.float32),axis=0)
            # if not keras
            float_image = tf.image.convert_image_dtype(distorted_image, dtype=tf.float32)
        return float_image, labels
