import numpy as np
import pandas as pd


data = pd.read_csv('pre_total.csv')
print(data.columns)


class DataProcess():
    def __init__(self):
        self.pre_total_pd = pd.read_csv('pre_total.csv')
        self.post_education_total_pd = pd.read_csv('post_education_total.csv')

    def merge_data(self):
        # post apn & rn merge
        post_education_apn_pd = pd.read_csv('post_education_APN.csv')
        post_education_rn_pd = pd.read_csv('post_education_RN.csv')

        post_education_total_pd = pd.concat([post_education_rn_pd, post_education_apn_pd], axis=0)

        # save post total data
        post_education_total_pd.to_csv('post_education_total.csv', index=False)

    @staticmethod
    def education_answer(number):
        if number == 1:
            return {'question': '1. According to American Academy of Pediatrics (AAP), what is the ideal sound level in NICU?', 'ans': 'Below 45 dB'}

        elif number == 2:
            pass

    def main_process(self):
        for num in [1]:
            question = self.education_answer(num)['question']
            answer = self.education_answer(num)['ans']

            print(self.pre_total_pd[question])


if __name__ == '__main__':
    data_process = DataProcess()
    data_process.main_process()
