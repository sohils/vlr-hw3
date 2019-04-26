from student_code.coattention_net import CoattentionNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset

import torch


class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
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
        

        self._model = CoattentionNet(vocab_size=len(train_dataset.word2idx_question_base), embedding_dim=512, max_len=train_dataset.max_question_len)
        
        self.criteron = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.RMSprop(self._model.parameters(), lr=4e-4, weight_decay=1e-8, momentum=0.99)

        super().__init__(train_dataset, val_dataset, self._model, batch_size, num_epochs,
                         num_data_loader_workers=num_data_loader_workers)

    def _optimize(self, predicted_answers, true_answer_ids, train=False):
        loss = self.criteron(predicted_answers, true_answer_ids.long())
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss
