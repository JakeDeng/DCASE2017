# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:25:23 2018

Custom_task1 : Scene Classification

@author: jake
"""

from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import numpy
import argparse
import textwrap
from tqdm import tqdm
from six import iteritems
import logging
import sed_eval
import platform
import pkg_resources
import warnings
import collections


from dcase_framework.application_core import AcousticSceneClassificationAppCore, SoundEventAppCore
from dcase_framework.utils import *
from dcase_framework.learners import EventDetector ,SceneClassifier
from dcase_framework.keras_utils import KerasMixin, BaseDataGenerator, StasherCallback
from dcase_framework.datasets import AcousticSceneDataset

from dcase_framework.containers import DottedDict
from dcase_framework.files import ParameterFile
from dcase_framework.features import FeatureContainer, FeatureRepository, FeatureExtractor, FeatureNormalizer, \
    FeatureStacker, FeatureAggregator, FeatureMasker
from dcase_framework.datasets import *
from dcase_framework.utils import filelist_exists, Timer
from dcase_framework.decorators import before_and_after_function_wrapper
from dcase_framework.learners import *
from dcase_framework.recognizers import SceneRecognizer, EventRecognizer
from dcase_framework.metadata import MetaDataContainer, MetaDataItem
from dcase_framework.ui import FancyLogger
from dcase_framework.utils import get_class_inheritors, posix_path, check_pkg_resources
from dcase_framework.parameters import ParameterContainer
from dcase_framework.files import ParameterFile
from dcase_framework.data import DataProcessor, ProcessingChain

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)

#add custom class and methods


class CustomFeatureExtractor(FeatureExtractor):
    pass

class CustomMLP(EventDetector, KerasMixin):

    """KerasMixin class using keras"""
       
    #constructor
    def __init__(self, *args, **kwargs):
        super(CustomMLP, self).__init__(*args, **kwargs)
        #self.method = 'mlp'

    #override the original learn method
    #learn method is used in class AcousticSceneClassificationAppCore-->system_training method
    def learn(self, data, annotations, data_filenames=None, validation_files=[],**kwargs):
        """
        Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data
        data_filenames : dict of filenames
            Filenames of stored data
        validation_files: list of filenames
            Predefined validation files, use parameter 'validation.setup_source=dataset' to use them.

        Returns
        -------
        self

        """

        #----------------------handle training files and validation files----------------------
        # Collect training files
        training_files = sorted(list(annotations.keys()))

        # Validation files
        if self.learner_params.get_path('validation.enable', False):
            if self.learner_params.get_path('validation.setup_source').startswith('generated'):
                validation_files = self._generate_validation(
                    annotations=annotations,
                    validation_type=self.learner_params.get_path('validation.setup_source', 'generated_scene_event_balanced'),
                    valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                    seed=self.learner_params.get_path('validation.seed'),
                )

            elif self.learner_params.get_path('validation.setup_source') == 'dataset':
                if validation_files:
                    validation_files = sorted(list(set(validation_files).intersection(training_files)))

                else:
                    message = '{name}: No validation_files set'.format(
                        name=self.__class__.__name__
                    )

                    self.logger.exception(message)
                    raise ValueError(message)

            else:
                message = '{name}: Unknown validation.setup_source [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.learner_params.get_path('validation.setup_source')
                )

                self.logger.exception(message)
                raise ValueError(message)

            training_files = sorted(list(set(training_files) - set(validation_files)))

        else:
            validation_files = []

        # Double check that training and validation files are not overlapping.
        if set(training_files).intersection(validation_files):
            message = '{name}: Training and validation file lists are overlapping!'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        #print out when it is done
        print('training_files :' ,training_files,'\n')
        print('validation_files :',validation_files,'\n')

        #----------------------handle training files and validation files----------------------

        #----------------------Process feature data and annotation matrix----------------------

        # Convert annotations into activity matrix format
        #_get_target_matrix_dict is in LearnerContainer Class
        activity_matrix_dict = self._get_target_matrix_dict(data=data, annotations=annotations)

        # Process data

        #prepare_data and prepare_activity are in KerasMixin Class
        #numpy.vstacking feature and annotations into one matrix
        #modify feature format if using CNN feature
        global X_training, Y_training
        X_training = self.prepare_data(data=data, files=training_files)
        Y_training = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=training_files)


        if self.show_extra_debug:
            self.logger.debug('  Training items \t[{examples:d}]'.format(examples=len(X_training)))

        # Process validation data
        if validation_files:
            X_validation = self.prepare_data(data=data, files=validation_files)
            Y_validation = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=validation_files)

            validation = (X_validation, Y_validation)
            if self.show_extra_debug:
                self.logger.debug('  Validation items \t[{validation:d}]'.format(validation=len(X_validation)))

        else:
            validation = None
        #----------------------Process feature data and annotation matrix----------------------

        #----------------------Set up environment and create Training Model----------------------
        # Set seed
        self.set_seed()
        # Setup keras backend and parameters
        # Setup Keras
        self._setup_keras()

        with SuppressStdoutAndStderr():
            # Import keras and suppress backend announcement printed to stderr
            import keras

        # Create model and optimizer and compile based on the model config
        # _get_input_size return input shape


        self.create_model(input_shape=self._get_input_size(data=data))

        if self.show_extra_debug:
            self.log_model_summary()

        class_weight = None
        if len(self.class_labels) == 1:
            # Special case with binary classifier
            if self.learner_params.get_path('training.class_weight'):
                class_weight = {}
                for class_id, weight in enumerate(self.learner_params.get_path('training.class_weight')):
                    class_weight[class_id] = float(weight)

            if self.show_extra_debug:
                negative_examples_id = numpy.where(Y_training[:, 0] == 0)[0]
                positive_examples_id = numpy.where(Y_training[:, 0] == 1)[0]

                self.logger.debug('  Positives items \t[{positives:d}]\t({percentage:.2f} %)'.format(
                    positives=len(positive_examples_id),
                    percentage=len(positive_examples_id)/float(len(positive_examples_id)+len(negative_examples_id))*100
                ))
                self.logger.debug('  Negatives items \t[{negatives:d}]\t({percentage:.2f} %)'.format(
                    negatives=len(negative_examples_id),
                    percentage=len(negative_examples_id) / float(len(positive_examples_id) + len(negative_examples_id)) * 100
                ))

                self.logger.debug('  Class weights \t[{weights}]\t'.format(weights=class_weight))

        callback_list = self.create_callback_list()

        #logging
        if self.show_extra_debug:
            self.logger.debug('  Feature vector \t[{vector:d}]'.format(
                vector=self._get_input_size(data=data))
            )
            self.logger.debug('  Batch size \t[{batch:d}]'.format(
                batch=self.learner_params.get_path('training.batch_size', 1))
            )

            self.logger.debug('  Epochs \t\t[{epoch:d}]'.format(
                epoch=self.learner_params.get_path('training.epochs', 1))
            )

        # Set seed
        self.set_seed()

        #model training
        hist = self.model.fit(
            x=X_training,
            y=Y_training,
            batch_size=self.learner_params.get_path('training.batch_size', 1),
            epochs=self.learner_params.get_path('training.epochs', 1),
            validation_data=validation,
            verbose=0,
            shuffle=self.learner_params.get_path('training.shuffle', True),
            callbacks=callback_list,
            class_weight=class_weight
        )

        # Manually update callbacks
        for callback in callback_list:
            if hasattr(callback, 'close'):
                callback.close()

        for callback in callback_list:
            if isinstance(callback, StasherCallback):
                callback.log()
                best_weights = callback.get_best()['weights']
                if best_weights:
                    self.model.set_weights(best_weights)
                break

        self['learning_history'] = hist.history
    #----------------------Set up environment and create Training Model----------------------
    
    #predict when training is done
    def predict(self, feature_data):

        if isinstance(feature_data, FeatureContainer):
            feature_data = feature_data.feat[0]

        return self.model.predict(x=feature_data).T


#cnn or rnn or crnn method
class CustomAppCore(SoundEventAppCore):
    def __init__(self, *args, **kwargs):
        #change model
        kwargs['Learners'] = {
            'crnn': CustomMLP,
        }

        #change extractor if needed
        kwargs['FeatureExtractor'] = CustomFeatureExtractor

        super(CustomAppCore, self).__init__(*args, **kwargs)

 
    #overwrite system_training method with CNN formatting
    @before_and_after_function_wrapper
    def system_training(self, overwrite=None):
        """System training stage

        Parameters
        ----------

        overwrite : bool
            overwrite existing models
            (Default value=False)

        Raises
        -------
        ValueError:
            Unknown scene_handling mode
        IOError:
            Feature normalizer not found.
            Feature file not found.

        Returns
        -------
        None

        """

        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)

        if self.params.get_path('learner.scene_handling') == 'scene-dependent':
            fold_progress = tqdm(
                self._get_active_folds(),
                desc='           {0:<15s}'.format('Fold '),
                file=sys.stdout,
                leave=False,
                miniters=1,
                disable=self.disable_progress_bar,
                ascii=self.use_ascii_progress_bar
            )

            for fold in fold_progress:
                if self.log_system_progress:
                    self.logger.info('  {title:<15s} [{fold:d}/{total:d}]'.format(
                        title='Fold',
                        fold=fold,
                        total=len(fold_progress))
                    )

                scene_labels = self.dataset.scene_labels
                # Select only active scenes
                if self.params.get_path('learner.active_scenes'):
                    scene_labels = list(
                        set(scene_labels).intersection(
                            self.params.get_path('learner.active_scenes')
                        )
                    )

                scene_progress = tqdm(
                    scene_labels,
                    desc="           {0: >15s}".format('Scene '),
                    file=sys.stdout,
                    leave=False,
                    miniters=1,
                    disable=self.disable_progress_bar,
                    ascii=self.use_ascii_progress_bar
                )

                for scene_label in scene_progress:
                    current_model_file = self._get_model_filename(
                        fold=fold,
                        path=self.params.get_path('path.learner'),
                        scene_label=scene_label
                    )

                    if not os.path.isfile(current_model_file) or overwrite:
                        feature_processing_chain = self.ProcessingChain()

                        # Feature stacker
                        feature_stacker = FeatureStacker(
                            recipe=self.params.get_path('feature_stacker.stacking_recipe'),
                            feature_hop=self.params.get_path('feature_stacker.feature_hop', 1)
                        )
                        feature_processing_chain.append(feature_stacker)

                        # Feature normalizer
                        feature_normalizer = None
                        if self.params.get_path('feature_normalizer.enable'):
                            # Load normalizers
                            feature_normalizer_filenames = self._get_feature_normalizer_filename(
                                fold=fold,
                                path=self.params.get_path('path.feature_normalizer'),
                                scene_label=scene_label
                            )

                            normalizer_list = {}
                            for method, feature_normalizer_filename in iteritems(feature_normalizer_filenames):
                                if os.path.isfile(feature_normalizer_filename):
                                    normalizer_list[method] = FeatureNormalizer().load(
                                        filename=feature_normalizer_filename
                                    )

                                else:
                                    message = '{name}: Feature normalizer not found [{file}]'.format(
                                        name=self.__class__.__name__,
                                        file=feature_normalizer_filename
                                    )

                                    self.logger.exception(message)
                                    raise IOError(message)

                            feature_normalizer = self.FeatureNormalizer(feature_stacker.normalizer(
                                normalizer_list=normalizer_list)
                            )
                            feature_processing_chain.append(feature_normalizer)

                        # Feature aggregator
                        feature_aggregator = None
                        if self.params.get_path('feature_aggregator.enable'):
                            feature_aggregator = FeatureAggregator(
                                recipe=self.params.get_path('feature_aggregator.aggregation_recipe'),
                                win_length_frames=self.params.get_path('feature_aggregator.win_length_frames'),
                                hop_length_frames=self.params.get_path('feature_aggregator.hop_length_frames')
                            )
                            feature_processing_chain.append(feature_aggregator)

                        # Data processing chain
                        data_processing_chain = self.ProcessingChain()
                        if self.params.get_path('learner.parameters.input_sequencer.enable'):
                            data_sequencer = self.DataSequencer(
                                frames=self.params.get_path('learner.parameters.input_sequencer.frames'),
                                hop=self.params.get_path('learner.parameters.input_sequencer.hop'),
                                padding=self.params.get_path('learner.parameters.input_sequencer.padding'),
                                shift_step=self.params.get_path('learner.parameters.temporal_shifter.step') if self.params.get_path('learner.parameters.temporal_shifter.enable') else None,
                                shift_border=self.params.get_path('learner.parameters.temporal_shifter.border') if self.params.get_path('learner.parameters.temporal_shifter.enable') else None,
                                shift_max=self.params.get_path('learner.parameters.temporal_shifter.max') if self.params.get_path('learner.parameters.temporal_shifter.enable') else None,
                            )
                            data_processing_chain.append(data_sequencer)

                        # Data processor
                        data_processor = self.DataProcessor(
                            feature_processing_chain=feature_processing_chain,
                            data_processing_chain=data_processing_chain,
                        )

                        # Collect training examples
                        train_meta = self.dataset.train(fold=fold, scene_label=scene_label)
                        data = {}
                        data_filelist = {}
                        annotations = {}

                        item_progress = tqdm(train_meta.file_list[::self.params.get_path('learner.file_hop', 1)],
                                             desc="           {0: >15s}".format('Collect data '),
                                             file=sys.stdout,
                                             leave=False,
                                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ',
                                             disable=self.disable_progress_bar,
                                             ascii=self.use_ascii_progress_bar)

                        #-------------------------------loading data and annotation dictionary-------------------------------
                        for item_id, audio_filename in enumerate(item_progress):
                            if self.log_system_progress:
                                self.logger.info(
                                    '  {title:<15s} [{item_id:3s}/{total:3s}] {item:<20s}'.format(
                                        title='Collect data ',
                                        item_id='{:d}'.format(item_id),
                                        total='{:d}'.format(len(item_progress)),
                                        item=os.path.split(audio_filename)[-1])
                                    )

                            item_progress.set_postfix(file=os.path.splitext(os.path.split(audio_filename)[-1])[0])
                            item_progress.update()

                            # Get feature filenames
                            feature_filenames = self._get_feature_filename(
                                audio_file=audio_filename,
                                path=self.params.get_path('path.feature_extractor')
                            )

                            if not self.params.get_path('learner.parameters.generator.enable'):
                                # If generator is not used, load features now.
                                # Do only feature processing here. Leave data processing for learner.

                                feature_data, feature_length = data_processor.load(
                                    feature_filename_dict=feature_filenames,
                                    process_features=True,
                                    process_data=False
                                )
                                data[audio_filename] = FeatureContainer(features=[feature_data])

                            # Inject audio_filename to the features filenames for the raw data generator
                            feature_filenames['_audio_filename'] = audio_filename
                            data_filelist[audio_filename] = feature_filenames

                            annotations[audio_filename] = train_meta.filter(filename=audio_filename)
                        #-------------------------------loading data and annotation dictionary-------------------------------

                        #-------------------------------CNN feature transformation-------------------------------
                        '''
                            feature reformation operates upon dict data, which contains all the training feature data
                        '''
                        #if ture then perform 2D + channel format
                        if self.params.get_path('2D_feature')['cnn_format']:
                            if isinstance( data, FeatureContainer): 
                                data.feat[0] =data.feat[0].reshape(data.feat[0].shape[0], self.params.get_path('2D_feature')['frames'], 40, 1)

                            #feature dictionary that contains several feature containers-->for system training
                            else:
                                for x in data.keys():
                                    data[x].feat[0] =data[x].feat[0].reshape(data[x].feat[0].shape[0],self.params.get_path('2D_feature')['frames'], 40, 1)
                        print('\n','feature data shape :',data[list(data.keys())[0]].shape)

                        #-------------------------------CNN feature transformation-------------------------------

                        #-------------------------------RNN feature transformation-------------------------------
                        if self.params.get_path('2D_feature')['rnn_format']:
                            if isinstance( data, FeatureContainer): 
                                data.feat[0] =data.feat[0].reshape(data.feat[0].shape[0], self.params.get_path('2D_feature')['frames'], 40)

                            #feature dictionary that contains several feature containers-->for system training
                            else:
                                for x in data.keys():
                                    data[x].feat[0] =data[x].feat[0].reshape(data[x].feat[0].shape[0],self.params.get_path('2D_feature')['frames'], 40)
                        print('\n','feature data shape :',data[list(data.keys())[0]].shape)

                        #-------------------------------RNN feature transformation-------------------------------

                        if self.log_system_progress:
                            self.logger.info(' ')

                        # Get learner
                        learner = self._get_learner(
                            method=self.params.get_path('learner.method'),
                            class_labels=self.dataset.event_labels(scene_label=scene_label),
                            data_processor=data_processor,
                            feature_processing_chain=feature_processing_chain,
                            feature_normalizer=feature_normalizer,
                            feature_stacker=feature_stacker,
                            feature_aggregator=feature_aggregator,
                            params=self.params.get_path('learner'),
                            filename=current_model_file,
                            disable_progress_bar=self.disable_progress_bar,
                            log_progress=self.log_system_progress,
                            data_generators=self.DataGenerators if self.params.get_path('learner.parameters.generator.enable') else None,
                        )

                        # Get validation files from dataset
                        validation_files = self.dataset.validation_files(fold=fold, scene_label=scene_label)

                        # Start learning
                        learner.learn(
                            data=data,
                            annotations=annotations,
                            data_filenames=data_filelist,
                            validation_files=validation_files
                        )

                        learner.save()

        elif self.params.get_path('learner.scene_handling') == 'scene-independent':
            message = '{name}: Scene handling mode not implemented yet [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('learner.scene_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)

        else:
            message = '{name}: Unknown scene handling mode [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('learner.scene_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)

    @before_and_after_function_wrapper
    def system_testing(self, overwrite=None):
        """System testing stage

        If extracted features are not found from disk, they are extracted but not saved.

        Parameters
        ----------
        overwrite : bool
            overwrite existing models
            (Default value=False)

        Raises
        -------
        ValueError:
            Unknown scene_handling mode
        IOError:
            Model file not found.
            Audio file not found.

        Returns
        -------
        None

        """

        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)

        if self.params.get_path('recognizer.scene_handling') == 'scene-dependent':

            fold_progress = tqdm(self._get_active_folds(),
                                 desc="           {0: >15s}".format('Fold '),
                                 file=sys.stdout,
                                 leave=False,
                                 miniters=1,
                                 disable=self.disable_progress_bar,
                                 ascii=self.use_ascii_progress_bar)

            for fold in fold_progress:
                if self.log_system_progress:
                    self.logger.info('  {title:<15s} [{fold:d}/{total:d}]'.format(
                        title='Fold',
                        fold=fold,
                        total=len(fold_progress))
                    )

                scene_labels = self.dataset.scene_labels
                # Select only active scenes
                if self.params.get_path('recognizer.active_scenes'):
                    scene_labels = list(
                        set(scene_labels).intersection(
                            self.params.get_path('recognizer.active_scenes')
                        )
                    )

                scene_progress = tqdm(scene_labels,
                                      desc="           {0: >15s}".format('Scene '),
                                      file=sys.stdout,
                                      leave=False,
                                      miniters=1,
                                      disable=self.disable_progress_bar,
                                      ascii=self.use_ascii_progress_bar)

                for scene_label in scene_progress:
                    current_result_file = self._get_result_filename(
                        fold=fold,
                        path=self.params.get_path('path.recognizer'),
                        scene_label=scene_label
                    )

                    if not os.path.isfile(current_result_file) or overwrite:
                        results = MetaDataContainer(filename=current_result_file)

                        # Load class model container
                        model_filename = self._get_model_filename(fold=fold,
                                                                  path=self.params.get_path('path.learner'),
                                                                  scene_label=scene_label
                                                                  )

                        if os.path.isfile(model_filename):
                            model_container = self._get_learner(method=self.params.get_path('learner.method')).load(
                                filename=model_filename)
                        else:
                            message = '{name}: Model file not found [{file}]'.format(
                                name=self.__class__.__name__,
                                file=model_filename
                            )

                            self.logger.exception(message)
                            raise IOError(message)

                        item_progress = tqdm(self.dataset.test(fold, scene_label=scene_label).file_list,
                                             desc="           {0: >15s}".format('Testing '),
                                             file=sys.stdout,
                                             leave=False,
                                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ',
                                             disable=self.disable_progress_bar,
                                             ascii=self.use_ascii_progress_bar
                                             )
                        for item_id, audio_filename in enumerate(item_progress):
                            if self.log_system_progress:
                                self.logger.info(
                                    '  {title:<15s} [{item_id:d}/{total:d}] {item:<20s}'.format(
                                        title='Testing',
                                        item_id=item_id,
                                        total=len(item_progress),
                                        item=os.path.split(audio_filename)[-1])
                                    )

                            # Load features
                            feature_filenames = self._get_feature_filename(
                                audio_file=audio_filename,
                                path=self.params.get_path('path.feature_extractor')
                            )

                            feature_list = {}
                            for method, feature_filename in iteritems(feature_filenames):
                                if os.path.isfile(feature_filename):
                                    feature_list[method] = FeatureContainer().load(filename=feature_filename)
                                else:
                                    message = '{name}: Features not found [{file}]'.format(
                                        name=self.__class__.__name__,
                                        file=audio_filename
                                    )

                                    self.logger.exception(message)
                                    raise IOError(message)

                            if hasattr(model_container, 'data_processor'):
                                # Leave feature and data processing to DataProcessor stored inside the model
                                feature_data = feature_list

                            else:
                                # Backward compatibility mode
                                feature_data = model_container.feature_stacker.process(
                                    feature_data=feature_list
                                )

                                # Normalize features
                                if model_container.feature_normalizer:
                                    feature_data = model_container.feature_normalizer.normalize(feature_data)

                                # Aggregate features
                                if model_container.feature_aggregator:
                                    feature_data = model_container.feature_aggregator.process(feature_data)

                            #-------------------------------CNN feature transformation-------------------------------
                            '''
                                feature reformation operates upon dict data, which contains all the training feature data
                            '''
                            #if ture then perform 2D format
                            if self.params.get_path('2D_feature')['cnn_format']:
                                if isinstance( feature_data, FeatureContainer): 
                                    feature_data.feat[0] =feature_data.feat[0].reshape(feature_data.feat[0].shape[0], self.params.get_path('2D_feature')['frames'], 40, 1)
                                
                            #-------------------------------CNN feature transformation-------------------------------
                            #-------------------------------RNN feature transformation-------------------------------
                            '''
                                feature reformation operates upon dict data, which contains all the training feature data
                            '''
                            #if ture then perform 2D format
                            if self.params.get_path('2D_feature')['rnn_format']:
                                if isinstance( feature_data, FeatureContainer): 
                                    feature_data.feat[0] =feature_data.feat[0].reshape(feature_data.feat[0].shape[0], self.params.get_path('2D_feature')['frames'], 40)
                                
                            #-------------------------------RNN feature transformation-------------------------------
                               
                            # Frame probabilities
                            frame_probabilities = model_container.predict(
                                feature_data=feature_data,
                            )

                            print('probability shape :',frame_probabilities.shape)
                            #shape : (6 , n)

                            # Event recognizer
                            current_result = self.EventRecognizer(
                                hop_length_seconds=model_container.params.get_path('hop_length_seconds'),
                                params=self.params.get_path('recognizer'),
                                class_labels=model_container.class_labels
                            ).process(
                                frame_probabilities=frame_probabilities
                            )

                            for event in current_result:
                                event.file = self.dataset.absolute_to_relative(audio_filename)
                                results.append(event)

                        # Save testing results
                        results.save()

        elif self.params.get_path('recognizer.scene_handling') == 'scene-independent':
            message = '{name}: Scene handling mode not implemented yet [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('recognizer.scene_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)

        else:
            message = '{name}: Unknown scene handling mode [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('recognizer.scene_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)

    @before_and_after_function_wrapper
    def system_evaluation(self):
        """System evaluation stage.

        Testing outputs are collected and evaluated.

        Parameters
        ----------

        Returns
        -------
        None

        Raises
        -------
        IOError
            Result file not found

        """
        if not self.dataset.reference_data_present:
            return '  No reference data available for dataset.'
        else:
            output = ''
            if self.params.get_path('evaluator.scene_handling') == 'scene-dependent':
                overall_metrics_per_scene = {}

                scene_labels = self.dataset.scene_labels

                # Select only active scenes
                if self.params.get_path('evaluator.active_scenes'):
                    scene_labels = list(
                        set(scene_labels).intersection(
                            self.params.get_path('evaluator.active_scenes')
                        )
                    )

                for scene_id, scene_label in enumerate(scene_labels):
                    if scene_label not in overall_metrics_per_scene:
                        overall_metrics_per_scene[scene_label] = {}

                    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
                        event_label_list=self.dataset.event_labels(scene_label=scene_label),
                        time_resolution=1.0,
                    )

                    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
                        event_label_list=self.dataset.event_labels(scene_label=scene_label),
                        evaluate_onset=True,
                        evaluate_offset=False,
                        t_collar=0.5,
                        percentage_of_length=0.5
                    )

                    for fold in self._get_active_folds():
                        result_filename = self._get_result_filename(fold=fold,
                                                                    scene_label=scene_label,
                                                                    path=self.params.get_path('path.recognizer'))

                        results = MetaDataContainer().load(filename=result_filename)

                        for file_id, audio_filename in enumerate(self.dataset.test(fold, scene_label=scene_label).file_list):

                            # Select only row which are from current file and contains only detected event
                            current_file_results = MetaDataContainer()
                            for result_item in results.filter(
                                    filename=posix_path(self.dataset.absolute_to_relative(audio_filename))
                            ):
                                if 'event_label' in result_item and result_item.event_label:
                                    current_file_results.append(result_item)

                            meta = MetaDataContainer()
                            for meta_item in self.dataset.file_meta(
                                    filename=posix_path(self.dataset.absolute_to_relative(audio_filename))
                            ):
                                if 'event_label' in meta_item and meta_item.event_label:
                                    meta.append(meta_item)

                            segment_based_metric.evaluate(
                                reference_event_list=meta,
                                estimated_event_list=current_file_results
                            )

                            event_based_metric.evaluate(
                                reference_event_list=meta,
                                estimated_event_list=current_file_results
                            )

                    overall_metrics_per_scene[scene_label]['segment_based_metrics'] = segment_based_metric.results()
                    overall_metrics_per_scene[scene_label]['event_based_metrics'] = event_based_metric.results()
                    if self.params.get_path('evaluator.show_details', False):
                        output += "  Scene [{scene}], Evaluation over {folds:d} folds\n".format(
                            scene=scene_label,
                            folds=self.dataset.fold_count
                        )

                        output += " \n"
                        output += segment_based_metric.result_report_overall()
                        output += segment_based_metric.result_report_class_wise()
                overall_metrics_per_scene = DottedDict(overall_metrics_per_scene)

                output += " \n"
                output += "  Overall metrics \n"
                output += "  =============== \n"
                output += "    {event_label:<17s} | {segment_based_fscore:7s} | {segment_based_er:7s} | {event_based_fscore:7s} | {event_based_er:7s} | \n".format(
                    event_label='Event label',
                    segment_based_fscore='Seg. F1',
                    segment_based_er='Seg. ER',
                    event_based_fscore='Evt. F1',
                    event_based_er='Evt. ER',
                )
                output += "    {event_label:<17s} + {segment_based_fscore:7s} + {segment_based_er:7s} + {event_based_fscore:7s} + {event_based_er:7s} + \n".format(
                    event_label='-' * 17,
                    segment_based_fscore='-' * 7,
                    segment_based_er='-' * 7,
                    event_based_fscore='-' * 7,
                    event_based_er='-' * 7,
                )
                avg = {
                    'segment_based_fscore': [],
                    'segment_based_er': [],
                    'event_based_fscore': [],
                    'event_based_er': [],
                }
                for scene_id, scene_label in enumerate(scene_labels):
                    output += "    {scene_label:<17s} | {segment_based_fscore:<7s} | {segment_based_er:<7s} | {event_based_fscore:<7s} | {event_based_er:<7s} | \n".format(
                        scene_label=scene_label,
                        segment_based_fscore="{:4.2f}".format(overall_metrics_per_scene.get_path(scene_label + '.segment_based_metrics.overall.f_measure.f_measure') * 100),
                        segment_based_er="{:4.2f}".format(overall_metrics_per_scene.get_path(scene_label + '.segment_based_metrics.overall.error_rate.error_rate')),
                        event_based_fscore="{:4.2f}".format(overall_metrics_per_scene.get_path(scene_label + '.event_based_metrics.overall.f_measure.f_measure') * 100),
                        event_based_er="{:4.2f}".format(overall_metrics_per_scene.get_path(scene_label + '.event_based_metrics.overall.error_rate.error_rate')),
                    )

                    avg['segment_based_fscore'].append(overall_metrics_per_scene.get_path(scene_label + '.segment_based_metrics.overall.f_measure.f_measure') * 100)
                    avg['segment_based_er'].append(overall_metrics_per_scene.get_path(scene_label + '.segment_based_metrics.overall.error_rate.error_rate'))
                    avg['event_based_fscore'].append(overall_metrics_per_scene.get_path(scene_label + '.event_based_metrics.overall.f_measure.f_measure') * 100)
                    avg['event_based_er'].append(overall_metrics_per_scene.get_path(scene_label + '.event_based_metrics.overall.error_rate.error_rate'))

                output += "    {scene_label:<17s} + {segment_based_fscore:7s} + {segment_based_er:7s} + {event_based_fscore:7s} + {event_based_er:7s} + \n".format(
                    scene_label='-' * 17,
                    segment_based_fscore='-' * 7,
                    segment_based_er='-' * 7,
                    event_based_fscore='-' * 7,
                    event_based_er='-' * 7,
                )
                output += "    {scene_label:<17s} | {segment_based_fscore:<7s} | {segment_based_er:<7s} | {event_based_fscore:<7s} | {event_based_er:<7s} | \n".format(
                    scene_label='Average',
                    segment_based_fscore="{:4.2f}".format(numpy.mean(avg['segment_based_fscore'])),
                    segment_based_er="{:4.2f}".format(numpy.mean(avg['segment_based_er'])),
                    event_based_fscore="{:4.2f}".format(numpy.mean(avg['event_based_fscore'])),
                    event_based_er="{:4.2f}".format(numpy.mean(avg['event_based_er'])),
                )

            elif self.params.get_path('evaluator.scene_handling') == 'scene-independent':
                message = '{name}: Scene handling mode not implemented yet [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.params.get_path('evaluator.scene_handling')
                )

                self.logger.exception(message)
                raise ValueError(message)

            else:
                message = '{name}: Unknown scene handling mode [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.params.get_path('evaluator.scene_handling')
                )

                self.logger.exception(message)
                raise ValueError(message)

            if self.params.get_path('evaluator.saving.enable'):
                filename = self.params.get_path('evaluator.saving.filename').format(
                    dataset_name=self.dataset.storage_name,
                    parameter_set=self.params['active_set'],
                    parameter_hash=self.params['_hash']
                )

                output_file = os.path.join(self.params.get_path('path.evaluator'), filename)

                output_data = {
                    'overall_metrics_per_scene': overall_metrics_per_scene,
                    'average': {
                        'segment_based_fscore': numpy.mean(avg['segment_based_fscore']),
                        'segment_based_er': numpy.mean(avg['segment_based_er']),
                        'event_based_fscore': numpy.mean(avg['event_based_fscore']),
                        'event_based_er': numpy.mean(avg['event_based_er']),
                    },
                    'parameters': dict(self.params)
                }
                ParameterFile(output_data, filename=output_file).save()

            return output







def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Task 3: Acoustic Event Detection
            ---------------------------------------------
                Author:  Jake Deng

            System description
                A system for acoustic scene classification, using DCASE 2017 Challenge evalution dataset.
                Features: mean and std of centroid + zero crossing rate inside 1 second non-overlapping segments
                Classifier: MLP

        '''))

    # Setup argument handling
    parser.add_argument('-m', '--mode',
                        choices=('dev', 'challenge'),
                        default=None,
                        help="Selector for system mode",
                        required=False,
                        dest='mode',
                        type=str)

    parser.add_argument('-p', '--parameters',
                        help='parameter file override',
                        dest='parameter_override',
                        required=False,
                        metavar='FILE',
                        type=argument_file_exists)

    parser.add_argument('-s', '--parameter_set',
                        help='Parameter set id',
                        dest='parameter_set',
                        required=False,
                        type=str)

    parser.add_argument("-n", "--node",
                        help="Node mode",
                        dest="node_mode",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_sets",
                        help="List of available parameter sets",
                        dest="show_set_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_datasets",
                        help="List of available datasets",
                        dest="show_dataset_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_parameters",
                        help="Show parameters",
                        dest="show_parameters",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_eval",
                        help="Show evaluated setups",
                        dest="show_eval",
                        action='store_true',
                        required=False)

    parser.add_argument("-o", "--overwrite",
                        help="Overwrite mode",
                        dest="overwrite",
                        action='store_true',
                        required=False)

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)

    # Parse arguments
    args = parser.parse_args()

    # Load default parameters from a file
    default_parameters_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),'parameters',
                                               os.path.splitext(os.path.basename(__file__))[0]+'.defaults.yaml')
    if args.parameter_set:
        parameters_sets = args.parameter_set.split(',')
    else:
        parameters_sets = [None]

    for parameter_set in parameters_sets:
        # Initialize ParameterContainer
        params = ParameterContainer(project_base=os.path.dirname(os.path.realpath(__file__)))

        # Load default parameters from a file
        params.load(filename=default_parameters_filename)

        if args.parameter_override:
            # Override parameters from a file
            params.override(override=args.parameter_override)

        if parameter_set:
            # Override active_set
            params['active_set'] = parameter_set

        # Process parameters
        params.process()

#---------------------------------parameters are loaded into parameter container------------------------------

        # Force overwrite
        if args.overwrite:
            params['general']['overwrite'] = True

        # Override dataset mode from arguments
        if args.mode == 'dev':
            # Set dataset to development
            params['dataset']['method'] = 'development'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        elif args.mode == 'challenge':
            # Set dataset to training set for challenge
            params['dataset']['method'] = 'challenge_train'
            params['general']['challenge_submission_mode'] = True
            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        if args.node_mode:
            params['general']['log_system_progress'] = True
            params['general']['print_system_progress'] = False

        # Force ascii progress bar under Windows console
        if platform.system() == 'Windows':
            params['general']['use_ascii_progress_bar'] = True

        # Setup logging
        setup_logging(parameter_container=params['logging'])

#---------------------------------Initialize application core for training------------------------------
        #params file contains all the parameters that are needed for appcore object
        app = CustomAppCore(name='DCASE 2017::Sound Event Detection / Custom System',
                            params=params,
                            system_desc=params.get('description'),
                            system_parameter_set_id=params.get('active_set'),
                            setup_label='Development setup',
                            log_system_progress=params.get_path('general.log_system_progress'),
                            show_progress_in_console=params.get_path('general.print_system_progress'),
                            use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
                            )

        # Show parameter set list and exit
        if args.show_set_list:
            params_ = ParameterContainer(
                project_base=os.path.dirname(os.path.realpath(__file__))
            ).load(filename=default_parameters_filename)

            if args.parameter_override:
                # Override parameters from a file
                params_.override(override=args.parameter_override)
            if 'sets' in params_:
                app.show_parameter_set_list(set_list=params_['sets'])

            return

        # Show dataset list and exit
        if args.show_dataset_list:
            app.show_dataset_list()
            return

        # Show system parameters
        if params.get_path('general.log_system_parameters') or args.show_parameters:
            app.show_parameters()

        # Show evaluated systems
        if args.show_eval:
            app.show_eval()
            return

        # Initialize application
        # ==================================================
        #if params['flow']['initialize']:
        #   app.initialize()

        # Extract features for all audio files in the dataset
        # ==================================================
        if params['flow']['extract_features']:
            app.feature_extraction()

        # Prepare feature normalizers
        # ==================================================
        if params['flow']['feature_normalizer']:
            app.feature_normalization()

        # System training
        # ==================================================
        if params['flow']['train_system']:
            app.system_training()

        # System evaluation
        if not args.mode or args.mode == 'dev':

            # System testing
            # ==================================================
            if params['flow']['test_system']:
                app.system_testing()

            # System evaluation
            # ==================================================
            if params['flow']['evaluate_system']:
                app.system_evaluation()

        # System evaluation in challenge mode
        elif args.mode == 'challenge':
            # Set dataset to testing set for challenge
            params['dataset']['method'] = 'challenge_test'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters('dataset')

            if params['general']['challenge_submission_mode']:
                # If in submission mode, save results in separate folder for easier access
                params['path']['recognizer'] = params.get_path('path.recognizer_challenge_output')

            challenge_app = CustomAppCore(name='DCASE 2017::Acoustic Scene Classification / Baseline System',
                                          params=params,
                                          system_desc=params.get('description'),
                                          system_parameter_set_id=params.get('active_set'),
                                          setup_label='Evaluation setup',
                                          log_system_progress=params.get_path('general.log_system_progress'),
                                          show_progress_in_console=params.get_path('general.print_system_progress'),
                                          use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
                                          )
            # Initialize application
            if params['flow']['initialize']:
                challenge_app.initialize()

            # Extract features for all audio files in the dataset
            if params['flow']['extract_features']:
                challenge_app.feature_extraction()

            # System testing
            if params['flow']['test_system']:
                if params['general']['challenge_submission_mode']:
                    params['general']['overwrite'] = True

                challenge_app.system_testing()

                if params['general']['challenge_submission_mode']:
                    challenge_app.ui.line(" ")
                    challenge_app.ui.line("Results for the challenge are stored at ["+params.get_path('path.recognizer_challenge_output')+"]")
                    challenge_app.ui.line(" ")

            # System evaluation if not in challenge submission mode
            if params['flow']['evaluate_system']:
                challenge_app.system_evaluation()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

    
