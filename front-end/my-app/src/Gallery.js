import React, { Component } from 'react';
import './App.css';

class Image extends Component {
    render() {
        return (
            <span className="image-container">
                <img src={this.props.src} alt={this.props.src} onClick={this.props.onClick} />
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
            .then(json => { json.copyImages = json.images.slice(); this.setState(json); })
            .catch(err => console.log(err));

        this.sort = this.sort.bind(this);
    }

    sort() {
        var copyImages = this.state.copyImages.slice();
        const distance = this.state.images[this.state.currentSelection][1];

        const result = copyImages.map((item, index) => [distance[index], item])
            .sort(([index1], [index2]) => index1 - index2)
            .map(([, item]) => item);

        //console.log(result);

        this.setState({ "images": result });
    }

    handleClick(key) {
        this.setState({"currentSelection":key});
        this.sort();
    }

    render() {
        var sortedDistances = [];
        if (this.state.images[0])
            sortedDistances = this.state.images[0][1].sort();

        const images = this.state.images.map((value, key) => {
            return (
                <Image key={key}
                    src={this.baseUrl + "/" + value[0]}
                    distance={sortedDistances[key]}
                    onClick={() => this.handleClick(key)} />
            );
        });
        return (
            <div className="wrapper">
                {images}
            </div>
        );
    }
}

export default Gallery;