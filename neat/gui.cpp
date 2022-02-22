#include "gui.h"

using namespace std;
using namespace sf;

gui::gui(int w, int h, bool no_gui) : w(w), h(h), graphics_enabled(!no_gui) {

    if (graphics_enabled) {
        window = new RenderWindow(VideoMode(w, h), "Pretty NEAT");
        //shape = new CircleShape(100.f);
        //shape->setFillColor(Color::Green);
    } 
    
}

void gui::run_game() {
    population Pop{};
#if defined _DEBUG
    Pop.test();
#endif
    if (graphics_enabled) {
        Clock clock; // starts the clock
        
        Int32 previous_frame = clock.getElapsedTime().asMilliseconds();
        while (window->isOpen())
        {
            Pop.run_one_evolution_step();

            Event event;
            while (window->pollEvent(event))
            {
                if (event.type == Event::Closed)
                    window->close();
            }
            if (clock.getElapsedTime().asMilliseconds() - previous_frame > 50) {
                window->clear();
                Pop.draw(window);
                //window->draw(*shape);
                window->display();
                previous_frame = clock.getElapsedTime().asMilliseconds();
            }
        }
    }
    else {
        cout << "no graphics mode implemented yet" << endl;
    }
}