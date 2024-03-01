Sobre o texto Shakespeare.txt, disponível na seção Arquivos no diretório datasets, realizar as seguintes operações em Python de pré-processamento textual:

1. Normalização
Realizar as seguintes ações de normalização no arquivo de entrada e gerar o arquivo de saída Shakespeare_Normalized.txt.

- Lower case reduction
- Accent and diacritic removal
- Canonicalizing of acronyms, currency, date and hyphenated words
- Punctuation removal (except currency and date).
- Special characters removal
 
2. Tokenização
Realizar cada uma das seguintes tokenizações no arquivo ShakespeareNormalized.txt e gerar o arquivo de saída Shakespeare_Normalized_TokenizedXX.txt, onde XX é o número da subtarefa. Por exemplo, o arquivo Shakespeare_Normalized_Tokenized01.txt é a saída do algoritmo 1 (White Space Tokenization):

- White Space Tokenization
- NLTK: Word Tokenizer
- NLTK: Tree Bank Tokenizer
- NLTK: Word Punctuation Tokenizer
- NLTK: Tweet Tokenizer
- NLTK: MWE Tokenizer
- TextBlob Word Tokenizer
- spaCy Tokenizer
- Gensim Word Tokenizer
- Keras Tokenization
Para facilitar, teste primeiramente o funcionamento do código usando o seguinte texto:

```
It's true, Ms. Martha Töpfer! $3.00 on 3/21/2023 in cash for an ice-cream in the U.S. market? :-( #Truth
```
3. Stop-words Removal
Realizar a remoção de stop-words do texto (apenas o da subtarefa 2 de tokenização), e gerar um arquivo de saída denominado Shakespeare_Normalized_Tokenized_StopWord.txt.

4. Text Lemmatization
Realizar a lematização do texto gerado na etapa anterior, utilizando o WordNet Lemmatizer e gerar um arquivo de saída denominado Shakespeare_Normalized_Tokenized_StopWord_Lemmatized.txt.

5. Text Stemming
Aplicar cada um dos seguintes stemmers no arquivo de entrada Shakespeare_Normalized_Tokenized_StopWord_Lemmatized.txt e gerar o arquivo de saída Shakespeare_Normalized_Tokenized_StopWord_Lemmatized_StemmingXX.txt, onde XX é o número da subtarefa. Por exemplo, o arquivo Shakespeare_Normalized_Tokenized_StopWord_Lemmatized_Stemming01.txt é a saída do algoritmo 1 (Porter Stemmer):

- Porter Stemmer
- Snowball Stemmer
 
6. Análise do Vocabulário
Comparar os vocabulários gerados por cada lematizador e stemmer, apresentando um arquivo CSV para cada um deles contendo:

- Token (raíz resultante)
- Número de ocorrências do token no documento resultante (lematizado ou com stemming)
- Tamanho em caracteres de cada token do vocabulário
Por exemplo, para o lematizador, gerar o arquivo Shakespeare_Vocabulary_Lemmatized.csv e para o Porter Stemmer gerar o arquivo Shakespeare_Vocabulary_Porter.csv.

Apresentar um documento final comparativo denominado Shakespeare_Vocabulary_Analysis.txt contendo, para cada lematizador e stemmer utilizado, o tamanho do vocabulário (número de tokens), o número médio de ocorrências e o tamanho médio dos tokens.