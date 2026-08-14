#include <windows.h>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        // Default alarm
        Beep(1000, 300);
        Beep(1200, 300);
        Beep(1000, 500);
        return 0;
    }

    std::string type = argv[1];

    if (type == "spoof") {
        // Stronger alarm for spoof
        for (int i = 0; i < 3; i++) {
            Beep(800, 200);
            Beep(1200, 200);
        }
    }
    else if (type == "fear") {
        // Different pattern for fear
        Beep(600, 400);
        Beep(900, 400);
        Beep(600, 600);
    }
    else {
        // Generic alarm
        Beep(1000, 400);
        Beep(1500, 400);
    }

    return 0;
}