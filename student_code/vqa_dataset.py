from torch.utils.data import Dataset
from external.vqa.vqa import VQA

from PIL import Image

import torch
import torchvision.transforms as transforms

import numpy as np

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """
    word2idx_question_base = None
    idx2word_question_base = None
    word2idx_answer_base = None
    idx2word_answer_base = None
    max_question_len = None

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern, base_dict=False, transform=None,
    image_feature_dir=None, image_feature_pattern=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self.vqa_api_handle = VQA(annotation_json_file_path, question_json_file_path)
        self.image_filename_pattern = image_filename_pattern
        self.image_dir = image_dir
        self.image_feature_dir = image_feature_dir
        self.image_feature_pattern = image_feature_pattern
        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
                ])
        else :
            self.transform = transform

        if base_dict:
            # Gather all questions
            self.questions_list = self.gather_questions()
            # Create a vocabulary of words
            self.word2idx_question ,self.idx2word_question, self.max_question_len = build_volcabulary(self.questions_list, 5216, True)

            self.answers_list = self.gather_answers()
            self.word2idx_answer, self.idx2word_answer, _ = build_volcabulary(self.answers_list, 1000, False)

            self.valid_annotations = self.ann_idx_to_consider(self.word2idx_answer)
            
            VqaDataset.word2idx_question_base = self.word2idx_question
            VqaDataset.word2idx_answer_base = self.word2idx_answer
            VqaDataset.max_question_len = self.max_question_len
        else:
            self.valid_annotations = self.ann_idx_to_consider(VqaDataset.word2idx_answer_base)

        # self.image_features = self.load_image_to_features(image_feature_dir,image_feature_pattern)


    def __len__(self):
        return len(self.valid_annotations)
        # return len(self.vqa_api_handle.dataset['annotations'])
        # raise NotImplementedError()

    def __getitem__(self, elem_idx):
        idx = self.valid_annotations[elem_idx]
        # Get image number based on ID.
        ann = self.vqa_api_handle.dataset['annotations'][idx]
        # Load image
        img_num = ann['image_id']
        if(self.image_feature_dir == None or self.image_feature_pattern == None):
            img_fileName = self.image_filename_pattern.format("%012d"%img_num)
            img_path = self.image_dir + '/' + img_fileName
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                img = self.transform(img)
                # img = transforms.ToTensor()(img)
        else:
            # try:
                # img = torch.from_numpy(self.image_features[elem_idx]).squeeze()
            # except IndexError:
            #     print(idx)
            # img = torch.from_numpy(self.image_features[elem_idx]).squeeze()
            img = torch.from_numpy(np.load(self.image_feature_dir + '/' + self.image_feature_pattern.format("%012d"%img_num)))
        
        # Get list of questions based on image number.
        question_id = ann['question_id']
        question = self.get_question_for(idx)
        question_indices = self.get_indices(question, VqaDataset.word2idx_question_base)
        # question_indices = [self.word2idx_question[word] for word in question.split()]
        question_vec = torch.zeros(len(VqaDataset.word2idx_question_base))
        question_vec[question_indices] = 1

        # Get annotations of the respective questions.
        answer = self.get_answer_for(idx)
        answer_indices = self.get_indices(answer, VqaDataset.word2idx_answer_base)
        # answer_indices = [self.word2idx_answer[word] for word in answer.split()]
        answer_vec = torch.zeros(len(VqaDataset.word2idx_answer_base))
        answer_vec[answer_indices] = 1

        question_indices = question_indices + (VqaDataset.max_question_len - len(question_indices))*[0]
        question_indices = torch.tensor(question_indices)
        item = {'image':img, 'question':question_vec, 'answer':answer_vec, 'question_idxs':question_indices, 'answer_idxs':answer_indices}
        
        return item
    
    def load_image_to_features(self, image_feature_dir, image_feature_pattern):
        features = []
        for idx in range(len(self.valid_annotations)):
            ann = self.vqa_api_handle.dataset['annotations'][idx]
            img_num = ann['image_id']
            img_feature_fileName = self.image_feature_pattern.format("%012d"%img_num)
            img_feature_path = self.image_feature_dir + '/' + img_feature_fileName
            feature = np.load(img_feature_path)
            features.append(feature)
        return features


    def convert_image_to_features():
        # leNet = googlenet.googlenet(pretrained=True, only_features=True)
        for idx in range(len(self.vqa_api_handle.dataset['annotations'])):
            ann = self.vqa_api_handle.dataset['annotations'][idx]
            # Load image
            img_num = ann['image_id']
            img_fileName = self.image_filename_pattern.format('000000'+str(img_num))
            img_path = self.image_dir + '/' + img_fileName
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                img = self.transform(img)
                # img = transform(img)

    def get_indices(self, sentence, dictionary):
        indices = []
        for word in sentence.split():
            if word in dictionary:
                indices.append(dictionary[word])
        return indices

    def get_question_for(self, idx):
        ann = self.vqa_api_handle.dataset['annotations'][idx]
        question_id = ann['question_id']
        question = self.vqa_api_handle.qqa[question_id]['question']
        question = question.lower().replace("?","")
        return question

    def gather_questions(self):
        questions = []
        for idx in range(len(self.vqa_api_handle.dataset['annotations'])):
            questions.append(self.get_question_for(idx))
        return questions

    def get_answer_for(self, idx):
        ann = self.vqa_api_handle.dataset['annotations'][idx]
        question_id = ann['question_id']
        # # If we want to use all the answers
        # list_of_answers = self.vqa_api_handle.qa[question_id]['answers']
        # answer = ""
        # for a in list_of_answers:
        #     if((a['answer_confidence'] == 'yes' or a['answer_confidence'] == 'maybe') and not a['answer'] in answer):
        #         answer = answer + str(a['answer']) + " "
        
        # Only multiple choice answer
        answer = self.vqa_api_handle.qa[question_id]['multiple_choice_answer']
        return answer

    def gather_answers(self):
        answers = []
        for idx in range(len(self.vqa_api_handle.dataset['annotations'])):
            answer = self.get_answer_for(idx)
            answers.append(answer)
        return answers

    def ann_idx_to_consider(self, ans_volabulary):
        valid_annotations = []
        for idx in range(len(self.vqa_api_handle.dataset['annotations'])):
            answer = self.get_answer_for(idx)
            if answer in ans_volabulary:
                valid_annotations.append(idx)
        return valid_annotations

def build_volcabulary(sentence_lists, max_elemets, withNone):
    max_len = 0
    tokenized_sentences = tokenize_sentences(sentence_lists)
    if(withNone):
        vocabulary = [None]
        vocabulary_count = [99999]
    else:
        vocabulary = []
        vocabulary_count = []
    for sentence in tokenized_sentences:
        if len(sentence)>max_len:
            max_len = len(sentence)
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
                vocabulary_count.append(1)
            else:
                vocabulary_count[vocabulary.index(token)] = vocabulary_count[vocabulary.index(token)] + 1
    
    if(max_elemets > 0):
        # Sort the vocabulary based on vocabulary_count
        vocabulary = [x for _,x in sorted(zip(vocabulary_count, vocabulary), reverse=True)]
        vocabulary = vocabulary[:max_elemets]

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    return word2idx, idx2word, max_len
        

def tokenize_sentences(sentences):
        tokens = [x.split() for x in sentences]
        return tokens 