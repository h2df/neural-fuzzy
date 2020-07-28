#ifndef GLOBALVARIABLES_H
#define GLOBALVARIABLES_H

//------------------------------------------

#define INPUT_NEURONS		16
#define HIDDEN_NEURONS		2 //12
#define OUTPUT_NEURONS		3

//------------------------------------------

enum Symbol{LETTER_A=0, LETTER_B=1, UNKNOWN=2};

struct LetterStructure{
    Symbol symbol;
    int outputs[OUTPUT_NEURONS];
    float f[INPUT_NEURONS];
};

extern LetterStructure letters[20001];
extern LetterStructure testPattern;

//extern int NUMBER_OF_PATTERNS;
const int NUMBER_OF_PATTERNS = 20000;
const int NUMBER_OF_TRAINING_PATTERNS = 16000;
const int NUMBER_OF_TEST_PATTERNS = 4000;
extern bool patternsLoadedFromFile;
extern int MAX_EPOCHS;
extern double LEARNING_RATE;
//extern int GLOBAL_COUNTER;

#endif // GLOBALVARIABLES_H
