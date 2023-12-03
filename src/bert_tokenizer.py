import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
import json
import ijson
from tqdm import tqdm
import argparse
from src.character_idx import CharacterIndexer

def get_impressions_from_csv(path):	
        df = pd.read_csv(path, engine="python")
        imp = df['Report Impression']
        imp = imp.str.strip()
        imp = imp.replace('\n',' ', regex=True)
        imp = imp.replace('\s+', ' ', regex=True)
        imp = imp.str.strip()
        return imp

def tokenize(impressions, tokenizer, charbert):
        # BERT tokenizer always removes '\n' before tokenize input text.
        # For example, tokenizer.tokenize('cardiomegaly\nno') -> ['card', '##iom', '##ega', '##ly', 'no']
        new_impressions = []
        print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")

        # model with CharacterBERT
        if charbert:
                indexer = CharacterIndexer()
                for i in tqdm(range(impressions.shape[0])):
                        tokenized_imp = tokenizer.basic_tokenizer.tokenize(impressions.iloc[i])
                        if tokenized_imp: #not an empty report
                                tokenized_imp = ['[CLS]', *tokenized_imp, '[SEP]']
                                res = indexer.as_padded_tensor([tokenized_imp]).tolist()
                                if len(res) > 512: #length exceeds maximum size
                                        print("report {} length bigger than 512".format(i))
                                        res = res[:511] + indexer.as_padded_tensor([['[SEP]']]).tolist()
                                new_impressions += res
                        else: #an empty report
                                new_impressions += indexer.as_padded_tensor([['[CLS]', '[SEP]']]).tolist()
        # model without CharacterBERT
        else: 
                for i in tqdm(range(impressions.shape[0])):
                        tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
                        if tokenized_imp: #not an empty report
                                res = tokenizer.encode_plus(tokenized_imp)['input_ids']
                                if len(res) > 512: #length exceeds maximum size
                                        print("report {} length bigger than 512".format(i))
                                        res = res[:511] + [tokenizer.sep_token_id]
                                new_impressions.append(res)
                        else: #an empty report
                                new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id]) 
        return new_impressions

def load_list(path):
        with open(path, 'r') as filehandle:
                impressions = json.load(filehandle)
                return impressions
    

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Tokenize radiology report impressions and save as a list.')
        parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                            help='path to csv containing reports. The reports should be \
                            under the \"Report Impression\" column')
        parser.add_argument('-o', '--output_path', type=str, nargs='?', required=True,
                            help='path to intended output file')
        parser.add_argument('--charbert', type=str, nargs='?', required=False,
                            help='condition for using CharacterBERT')
        args = parser.parse_args()
        csv_path = args.data
        out_path = args.output_path
        charbert = args.charbert
        
        # use BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # get raw reports
        impressions = get_impressions_from_csv(csv_path)
        # tokenize reports
        new_impressions = tokenize(impressions, tokenizer, charbert)
        # save tokenized reports to new file
        with open(out_path, 'w') as filehandle:
                json.dump(new_impressions, filehandle)
