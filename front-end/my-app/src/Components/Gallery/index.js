import React from 'react';
import './Gallery.css';

import Image from '../Image';
import { BASE_URL, LIST_JSON_ROUTE } from '../../Constants/urls';

class Gallery extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      images: [],
      copyImages: [],
      sortedDistances: [],
    };

    // Currently selected image
    this.currentSelection = 0;
  }

  async componentDidMount() {
    const response = await fetch(`${BASE_URL}${LIST_JSON_ROUTE}`, { mode: 'cors' });
    const json = await response.json();

    this.setState({
      images: json.images,
      copyImages: json.images.slice(),
      sortedDistances: json.images[0][1],
    });
  }

  sortImages = () => {
    const distance = this.state.images[this.currentSelection][1];

    // Sort images by distance from currentSelection
    const result = this.state.copyImages
      .map((item, index) => [distance[index], item])
      .sort(([index1], [index2]) => index1 - index2);

    // Extract images and distances into two arrays
    const images = result.map(([, image]) => image);
    const sortedDistances = result.map(([sortedDistance]) => sortedDistance);

    this.setState({
      images,
      sortedDistances,
    });
  }

  handleClick(key) {
    this.currentSelection = key;
    this.sortImages();
  }

  ImageList = () =>
    this.state.images.map((value, key) => {
      const imageKey = `img_${key}`;
      return (
        <Image
          key={imageKey}
          src={`${BASE_URL}/${value[0]}`}
          distance={this.state.sortedDistances[key]}
          onClick={() => this.handleClick(key)}
        />
      );
    })

  render() {
    return (
      <div className="Gallery-wrapper">
        <this.ImageList />
      </div>
    );
  }
}

export default Gallery;
