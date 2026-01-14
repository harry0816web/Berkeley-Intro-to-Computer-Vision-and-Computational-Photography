# CS180 Computer Vision Project Website

A React-based website for showcasing CS180 Computer Vision assignments with a clean, modern design.

## Features

### 🎯 Assignment Header Component
- **Main Image Display**: Showcase your project's main result image
- **Title & Abstract**: Clear project title and detailed description
- **Assignment Number**: Easy identification of which assignment
- **Responsive Design**: Looks great on all devices

### 📊 Method Component
- **Tabbed Interface**: Switch between "Method Explanation" and "Results"
- **Method Explanation**: Detailed technical description with formatting support
- **Results Gallery**: Display multiple results with:
  - Result images
  - Descriptions
  - Performance metrics
  - Hover effects and animations

### 🔗 Footer Component
- **Social Links**: GitHub, Email, LinkedIn with icons
- **Professional Styling**: Clean, modern footer design
- **Responsive**: Adapts to mobile devices

## Getting Started

### 1. Install Dependencies
```bash
npm install
```

### 2. Add Your Images
Place your result images in the `public/` folder:
- `result.jpg` - Main showcase image
- `cathedral_result.jpg` - Cathedral alignment result
- `monastery_result.jpg` - Monastery alignment result  
- `tobolsk_result.jpg` - Tobolsk alignment result

### 3. Customize Your Content

#### Update Personal Information
In `src/App.jsx`, modify the Footer props:
```jsx
<Footer
  name="Your Name"
  githubLink="https://github.com/yourusername"
  email="your.email@example.com"
  linkedinLink="https://linkedin.com/in/yourprofile"
/>
```

#### Update Assignment Content
Modify the `assignmentData` object in `src/App.jsx`:
- Change the title and abstract
- Update the method explanation
- Add your actual results with real images and metrics
- Modify the assignment number

#### Example Result Object
```jsx
{
  title: "Your Result Title",
  image: "/your_image.jpg",
  description: "Description of your result",
  metrics: {
    "Metric 1": "Value 1",
    "Metric 2": "Value 2"
  }
}
```

### 4. Run the Development Server
```bash
npm run dev
```

### 5. Build for Production
```bash
npm run build
```

## Component Structure

```
src/
├── components/
│   ├── AssignmentHeader.jsx    # Header with title, image, abstract
│   ├── AssignmentHeader.css    # Header styling
│   ├── Method.jsx             # Method explanation and results
│   ├── Method.css             # Method component styling
│   ├── Footer.jsx             # Footer with social links
│   ├── Footer.css             # Footer styling
│   └── index.js               # Component exports
├── App.jsx                    # Main app with sample data
├── App.css                    # Global styles
└── main.jsx                   # React entry point
```

## Customization Tips

### Adding New Assignments
1. Create a new data object similar to `assignmentData`
2. Update the component props with your new content
3. Add your result images to the `public/` folder

### Styling Modifications
- Each component has its own CSS file for easy customization
- Uses CSS Grid and Flexbox for responsive layouts
- Gradient backgrounds and modern design elements
- Hover effects and smooth transitions

### Adding More Result Metrics
The results array supports any number of metrics:
```jsx
metrics: {
  "Accuracy": "95.2%",
  "Processing Time": "2.3s",
  "SSIM Score": "0.89",
  "Custom Metric": "Your Value"
}
```

## Deployment

### GitHub Pages
1. Build the project: `npm run build`
2. Deploy the `dist/` folder to GitHub Pages
3. Update image paths if needed

### Netlify/Vercel
1. Connect your GitHub repository
2. Set build command: `npm run build`
3. Set publish directory: `dist`

## Browser Support
- Modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile responsive design
- CSS Grid and Flexbox support required

## License
MIT License - Feel free to use and modify for your projects!
