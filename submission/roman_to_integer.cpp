#include <iostream>
#include <unordered_map>
#include <string>
using namespace std;

/*
    PROBLEM:
    Given a Roman numeral as a string, convert it to an integer.

    Example 1:
        Input:  "III"
        Output: 3

    Example 2:
        Input:  "IV"
        Output: 4

    Example 3:
        Input:  "MCMXCIV"
        Output: 1994

    ROMAN NUMERAL VALUES:
        I = 1
        V = 5
        X = 10
        L = 50
        C = 100
        D = 500
        M = 1000

    IDEA:
    - Traverse the Roman numeral string from left to right.
    - For each character, compare its value with the value of the next character.
    - If the current value is smaller than the next value, subtract the current value.
      This handles cases such as:
            IV = 4
            IX = 9
            XL = 40
            XC = 90
            CD = 400
            CM = 900

    - Otherwise, add the current value to the result.

    EXAMPLE WALKTHROUGH:
        s = "MCMXCIV"

        M = 1000, next is C = 100
            1000 >= 100, so add 1000

        C = 100, next is M = 1000
            100 < 1000, so subtract 100

        M = 1000, next is X = 10
            1000 >= 10, so add 1000

        X = 10, next is C = 100
            10 < 100, so subtract 10

        C = 100, next is I = 1
            100 >= 1, so add 100

        I = 1, next is V = 5
            1 < 5, so subtract 1

        V = 5, no next character
            add 5

        Final result:
            1000 - 100 + 1000 - 10 + 100 - 1 + 5 = 1994

    TIME COMPLEXITY:
        O(n), where n is the length of the Roman numeral string.
        We traverse the string only once.

    SPACE COMPLEXITY:
        O(1), because the map stores only 7 fixed Roman numeral characters.
*/

int romanToInt(string s) {
    unordered_map<char, int> romanValue = {
        {'I', 1},
        {'V', 5},
        {'X', 10},
        {'L', 50},
        {'C', 100},
        {'D', 500},
        {'M', 1000}
    };

    int result = 0;

    for (int i = 0; i < s.length(); i++) {
        int currentValue = romanValue[s[i]];

        if (i + 1 < s.length()) {
            int nextValue = romanValue[s[i + 1]];

            if (currentValue < nextValue) {
                result -= currentValue;
            } else {
                result += currentValue;
            }
        } else {
            result += currentValue;
        }
    }

    return result;
}

int main() {
    string roman;

    cout << "Enter a Roman numeral: ";
    cin >> roman;

    int decimal = romanToInt(roman);

    cout << "Decimal value: " << decimal << endl;

    return 0;
}