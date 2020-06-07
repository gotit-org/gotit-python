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

typedef struct colorPixel
{
    char *name;
    int red;
    int green;
    int blue;
} colorPixel;

vector<colorPixel> colors;

void InitColorsMap()
{
    colors.push_back({"Red", 255, 0, 0});
    colors.push_back({"Red", 165, 42, 42});
    colors.push_back({"Red", 139, 0, 0});
    colors.push_back({"Red", 220, 20, 60});
    colors.push_back({"Green", 0, 255, 0});
    colors.push_back({"Green", 0, 128, 0});
    colors.push_back({"Green", 0, 250, 154});
    colors.push_back({"Green", 154, 205, 50});
    colors.push_back({"Green", 173, 255, 47});
    colors.push_back({"Green", 127, 255, 0});
    colors.push_back({"Green", 34, 139, 34});
    colors.push_back({"Blue", 0, 0, 255});
    colors.push_back({"Blue", 65, 105, 225});
    colors.push_back({"Blue", 30, 144, 225});
    colors.push_back({"Blue", 0, 191, 225});
    colors.push_back({"Blue", 176, 224, 230});
    colors.push_back({"Brown", 160, 82, 45});
    colors.push_back({"Brown", 139, 69, 19});
    colors.push_back({"Yellow", 255, 255, 0});
    colors.push_back({"Yellow", 255, 215, 0});
    colors.push_back({"Orange", 255, 165, 0});
    colors.push_back({"Purple", 128, 0, 128});
    colors.push_back({"Purple", 186, 85, 211});
    colors.push_back({"Pink", 255, 105, 188});
    colors.push_back({"White", 255, 250, 250});
    colors.push_back({"White", 255, 255, 255});
    colors.push_back({"Gray", 169, 169, 169});
    colors.push_back({"Black", 0, 0, 0});
}

char *NearsetColor(unsigned char *pixel)
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
    return colors[colorIndex].name;
}

py::list GetColors(py::array_t<uint8_t> image, py::array_t<bool> mask)
{
    try
    {
        InitColorsMap();
        unsigned long long height = image.shape(0);
        unsigned long long width = image.shape(1);
        unsigned long long N = mask.shape(2);
        unsigned char *imagePointer = (unsigned char *)image.data();
        bool *maskPointer = (bool *)mask.data();
        
        string name;
        py::list result;
        vector<map<string, int>> colorsNames;
        colorsNames.resize(N);
        int *objectsSize = new int[N];
        memset(objectsSize, 0, sizeof(objectsSize));
        
        for (unsigned long long i = 0; i < height; i++)
        {
            for (unsigned long long j = 0; j < width; j++)
            {
                unsigned long long redIdx = (i * width + j) * 3;
                unsigned long long greenIdx = (i * width + j) * 3 + 1;
                unsigned long long blueIdx = (i * width + j) * 3 + 2;
                for(int k = 0; k < N; k++){
                    bool f = false;
                    unsigned long long idx = (i * width + j) * N + k;
                    if(maskPointer[idx]){
                        if(!f)
                            name = NearsetColor(new unsigned char[3]{imagePointer[redIdx], imagePointer[greenIdx], imagePointer[blueIdx]});
                        colorsNames[k][name]++;
                        objectsSize[k]++;
                    }
                }
            }
        }
        
        for(int i = 0; i < N; i++){
            py::list obj;
            for(auto key_value : colorsNames[i]){
                long double value = (long double)((int)(((key_value.second * 1.00)/objectsSize[i]) * 10000 + .5)) / 100.0;
                if(value >= 1.0){
                    py::dict color;
                    color["name"] = key_value.first;
                    color["percentage"] = value;
                    obj.append(color);
                }
            }
            result.append(obj);
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
