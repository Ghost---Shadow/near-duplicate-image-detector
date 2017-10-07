import React, { Component } from 'react';
import './App.css';

class Image extends Component {
    render() {
        return (
            <span className="image-container">
                <img src={this.props.src} alt={this.props.src} />
                <span>{this.props.distance}</span>
            </span>
        );
    }
}

class Gallery extends Component {
    constructor(props) {
        super(props);
        this.state = {
            "images": [],
            "currentSelection": 0
        }
        this.baseUrl = "http://localhost:8080";

        fetch(this.baseUrl + "/list.json", { mode: "cors" })
            .then(resp => resp.json())
            .then(json => { this.setState(json); this.state.copyImages = this.state.images.slice() })
            .catch(err => console.log(err));

        this.sort = this.sort.bind(this);
    }

    sort() {
        var copyImages = this.state.copyImages.slice();
        const distance = this.state.copyImages[this.state.currentSelection][1];

        const result = copyImages.map((item, index) => [distance[index], item])
            .sort(([index1], [index2]) => index1 - index2)
            .map(([, item]) => item);

        console.log(result);

        this.setState({ "images": result });
    }

    render() {
        const images = this.state.images.map((value, key) => {
            return (
                <Image key={key}
                    src={this.baseUrl + "/" + value[0]}
                    distance={value[1][this.state.currentSelection]} />
            );
        });
        return (
            <div className="wrapper">
                <button onClick={this.sort}>Sort</button>
                {images}
            </div>
        );
    }
}

export default Gallery;