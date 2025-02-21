# pyqt5-gui-app/pyqt5-gui-app/README.md

# PyQt5 GUI Application

This project is a PyQt5 GUI application that demonstrates a simple home page with two buttons: RGB Feed and Thermal Feed. The application follows the Model-View-Controller (MVC) architecture, separating the concerns of data management, user interface, and application logic.

## Project Structure

```
pyqt6-gui-app
├── controllers
│   ├── __init__.py
│   ├── home_controller.py
├── models
│   ├── __init__.py
│   ├── feed_model.py
├── views
│   ├── __init__.py
│   ├── home_view.py
├── main.py
└── README.md
```

## Components

### Controllers
- **home_controller.py**: Manages the interaction between the view and the model. It handles button clicks for RGB feed and Thermal feed.

### Models
- **feed_model.py**: Contains the data and logic related to the RGB and Thermal feeds.

### Views
- **home_view.py**: Creates the GUI for the home page, including buttons for RGB feed and Thermal feed.

### Main
- **main.py**: The entry point of the application that initializes the application, creates instances of the model, view, and controller, and starts the main event loop.

## Requirements

Ensure you have PyQt5 installed in your environment to run the application. You can install it using pip:

```
pip install PyQt5
```

## Usage

To run the application, execute the `main.py` file:

```
python main.py
```

This will launch the GUI, allowing you to interact with the RGB and Thermal feed buttons.

## Additional Notes

This project serves as a foundation for further development and can be expanded with additional features and functionalities as needed.