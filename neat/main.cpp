#include "gui.h"
std::random_device rdm;
std::uniform_real_distribution<> uniform;

int main() {
    std::cout << "started" << std::endl;

    uniform = std::uniform_real_distribution<>(0, 1);
    
    gui interface = gui(1280, 720);
    interface.run_game();

    std::cout << "done" << std::endl;
}   