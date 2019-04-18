from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset

import torch

class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path, train_image_feature_dir,
                 test_image_dir,test_question_path,test_annotation_path, test_image_feature_dir,
                 batch_size, num_epochs,
                 num_data_loader_workers):

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   image_feature_pattern="COCO_train2014_{}.npy",
                                   base_dict=True, image_feature_dir=train_image_feature_dir)
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 image_feature_pattern="COCO_val2014_{}.npy", 
                                 image_feature_dir=test_image_feature_dir)
        
        question_dict_size = len(train_dataset.word2idx_question_base)
        answer_dict_size = len(train_dataset.word2idx_answer_base)
        word_feature_size = 1024

        if(train_image_feature_dir == None):
            model = SimpleBaselineNet(question_dict_size, answer_dict_size, word_feature_size)
        else:
            model = SimpleBaselineNet(question_dict_size, answer_dict_size, word_feature_size, True)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD([{'params': model.question_embedding.parameters(), 'lr': 0.8},
                {'params': model.answering.parameters()}],lr=1e-2, momentum=0.9)

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)

    def _optimize(self, predicted_answers, true_answer_ids, train=False):
        # values, indices = true_answer_ids.max(1)
        # print(true_answer_ids.long())
        loss = self.criterion(predicted_answers, true_answer_ids.long())
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss
        
