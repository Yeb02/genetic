#include "population.h"

class gui {

public:
	int w, h;
	bool graphics_enabled;

private:
	sf::RenderWindow* window;
	sf::CircleShape* shape;


public:
	gui(int w = 800, int h = 600, bool no_gui = false);
	void run_game();
};