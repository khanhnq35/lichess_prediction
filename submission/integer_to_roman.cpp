#include <iostream>
#include <vector>
#include <string>
using namespace std;

/*
    PROBLEM:
    Given an integer, convert it to a Roman numeral.

    The input integer is usually in the range:
        1 <= num <= 3999

    Example 1:
        Input:  3
        Output: "III"

    Example 2:
        Input:  58
        Output: "LVIII"

        Explanation:
            L = 50
            V = 5
            III = 3
            58 = 50 + 5 + 3 = "LVIII"

    Example 3:
        Input:  1994
        Output: "MCMXCIV"

        Explanation:
            M = 1000
            CM = 900
            XC = 90
            IV = 4
            1994 = 1000 + 900 + 90 + 4 = "MCMXCIV"

    ROMAN NUMERAL VALUES:
        I  = 1
        IV = 4
        V  = 5
        IX = 9
        X  = 10
        XL = 40
        L  = 50
        XC = 90
        C  = 100
        CD = 400
        D  = 500
        CM = 900
        M  = 1000

    IDEA:
    - Use a greedy approach.
    - Store Roman numeral values from largest to smallest.
    - For each value, repeatedly subtract it from num while num is still large enough.
    - Every time we subtract a value, append the corresponding Roman symbol to the answer.

    WHY GREEDY WORKS:
    - Roman numerals are built by always using the largest possible valid symbol first.
    - Special cases such as 4, 9, 40, 90, 400, and 900 are included in the value table.
    - Therefore, we can safely choose the largest possible Roman value at each step.

    EXAMPLE WALKTHROUGH:
        num = 1994

        Start with result = ""

        1994 >= 1000:
            append "M"
            num = 1994 - 1000 = 994
            result = "M"

        994 >= 900:
            append "CM"
            num = 994 - 900 = 94
            result = "MCM"

        94 >= 90:
            append "XC"
            num = 94 - 90 = 4
            result = "MCMXC"

        4 >= 4:
            append "IV"
            num = 4 - 4 = 0
            result = "MCMXCIV"

        Final answer:
            "MCMXCIV"

    TIME COMPLEXITY:
        O(1)

        Although there is a loop, the input range is fixed from 1 to 3999.
        The number of Roman symbols is also fixed.

    SPACE COMPLEXITY:
        O(1)

        The value-symbol table has a fixed size.
        The output string size is also bounded because the input range is fixed.
*/

string intToRoman(int num) {
    vector<pair<int, string>> romanValues = {
        {1000, "M"},
        {900,  "CM"},
        {500,  "D"},
        {400,  "CD"},
        {100,  "C"},
        {90,   "XC"},
        {50,   "L"},
        {40,   "XL"},
        {10,   "X"},
        {9,    "IX"},
        {5,    "V"},
        {4,    "IV"},
        {1,    "I"}
    };

    string result = "";

    for (int i = 0; i < romanValues.size(); i++) {
        int value = romanValues[i].first;
        string symbol = romanValues[i].second;

        while (num >= value) {
            result += symbol;
            num -= value;
        }
    }

    return result;
}

int main() {
    int num;

    cout << "Enter an integer: ";
    cin >> num;

    string roman = intToRoman(num);

    cout << "Roman numeral: " << roman << endl;

    return 0;
}