#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <math.h>
#include <exception>

using namespace std;
namespace py = pybind11;
const int NUMBER_OF_COLORS = 11;

typedef struct colorPixel
{
    int color_id;
    char *name;
    int red;
    int green;
    int blue;
} colorPixel;

vector<colorPixel> colors;

void InitColorsMap()
{
    colors.push_back({0, "Red", 255, 0, 0});
    colors.push_back({0, "Red", 165, 42, 42});
    colors.push_back({0, "Red", 139, 0, 0});
    colors.push_back({0, "Red", 220, 20, 60});
    colors.push_back({1, "Green", 0, 255, 0});
    colors.push_back({1, "Green", 0, 128, 0});
    colors.push_back({1, "Green", 0, 250, 154});
    colors.push_back({1, "Green", 154, 205, 50});
    colors.push_back({1, "Green", 173, 255, 47});
    colors.push_back({1, "Green", 127, 255, 0});
    colors.push_back({1, "Green", 34, 139, 34});
    colors.push_back({2, "Blue", 0, 0, 255});
    colors.push_back({2, "Blue", 65, 105, 225});
    colors.push_back({2, "Blue", 30, 144, 225});
    colors.push_back({2, "Blue", 0, 191, 225});
    colors.push_back({2, "Blue", 176, 224, 230});
    colors.push_back({3, "Brown", 160, 82, 45});
    colors.push_back({3, "Brown", 139, 69, 19});
    colors.push_back({4, "Yellow", 255, 255, 0});
    colors.push_back({4, "Yellow", 255, 215, 0});
    colors.push_back({5, "Orange", 255, 165, 0});
    colors.push_back({6, "Purple", 128, 0, 128});
    colors.push_back({6, "Purple", 186, 85, 211});
    colors.push_back({7, "Pink", 255, 105, 188});
    colors.push_back({8, "White", 255, 250, 250});
    colors.push_back({8, "White", 255, 255, 255});
    colors.push_back({9, "Gray", 169, 169, 169});
    colors.push_back({10, "Black", 0, 0, 0});
}

int NearsetColor(unsigned char *pixel)
{
    int mindiff = INT_MAX, diff;
    int colorIndex = 0, redDiff = 0, greenDiff = 0, blueDiff = 0;
    unsigned long long NUMBER_OF_COLORS = colors.size();
    for (int i = 0; i < NUMBER_OF_COLORS; i++)
    {
        redDiff = abs(colors[i].red - pixel[0]);
        greenDiff = abs(colors[i].green - pixel[1]);
        blueDiff = abs(colors[i].blue - pixel[2]);
        diff = (redDiff + greenDiff + blueDiff) * 256;
        if (mindiff == INT_MAX || diff < mindiff)
        {
            mindiff = diff;
            colorIndex = i;
        }
    }
    return colors[colorIndex].color_id;
}

vector<vector<double>> GetColors(py::array_t<uint8_t> image, py::array_t<bool> mask)
{
    try
    {
        InitColorsMap();
        unsigned long long height = image.shape(0);
        unsigned long long width = image.shape(1);
        unsigned long long N = mask.shape(2);
        unsigned char *imagePointer = (unsigned char *)image.data();
        bool *maskPointer = (bool *)mask.data();
        int id;

        vector<vector<double>> result(N, vector<double>(NUMBER_OF_COLORS, 0));
        vector<vector<int>> colorsFreq(N, vector<int>(NUMBER_OF_COLORS, 0));
        vector<int> objectsSize(N, 0);

        for (unsigned long long i = 0; i < height; i++)
        {
            for (unsigned long long j = 0; j < width; j++)
            {
                unsigned long long redIdx = (i * width + j) * 3;
                unsigned long long greenIdx = (i * width + j) * 3 + 1;
                unsigned long long blueIdx = (i * width + j) * 3 + 2;
                bool f = false;
                for(int k = 0; k < N; k++){
                    unsigned long long idx = (i * width + j) * N + k;
                    if(maskPointer[idx]){
                        if(!f){
                            id = NearsetColor(new unsigned char[3]{imagePointer[redIdx], imagePointer[greenIdx], imagePointer[blueIdx]});
                            f = true;
                        }
                        colorsFreq[k][id]++;
                        objectsSize[k]++;
                    }
                }
            }
        }
        
        for(int i = 0; i < N; i++){
            for(int j = 0; j < NUMBER_OF_COLORS; j++){
                result[i][j] = (double)((int)(((colorsFreq[i][j] * 1.0)/objectsSize[i]) * 10000 + .5)) / 100.0;
            }
        }

        return result;
    }
    catch (exception& e)
    {
        py::print(e.what());
    }
}

PYBIND11_MODULE(ColorPy, m)
{
    m.doc() = "C++ code for color extraction"; // optional module docstring

    m.def("get_colors", &GetColors, "A function which detect colors from images");
}
