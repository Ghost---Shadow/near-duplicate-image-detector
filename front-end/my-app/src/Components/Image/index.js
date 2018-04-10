import React from 'react';
import PropTypes from 'react-proptypes';

import './Image.css';

const Image = props => (
  <span className="Image-container">
    <button
      className="Image-button"
      onClick={props.onClick}
    >
      <img
        className="Image-img"
        src={props.src}
        alt={props.src}
      />
      <span>{props.distance}</span>
    </button>
  </span>
);

Image.propTypes = {
  onClick: PropTypes.func.isRequired,
  src: PropTypes.string.isRequired,
  distance: PropTypes.number.isRequired,
};

export default Image;
