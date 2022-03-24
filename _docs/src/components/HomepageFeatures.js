import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Relevant topics',
    Svg: require('../../static/img/front-eagle.svg').default,
    description: (
      <>
        Documentation is very detailed and covers a wide range of topics related to recommender systems.
      </>
    ),
  },
  {
    title: 'Helps you discover',
    Svg: require('../../static/img/front-fly.svg').default,
    description: (
      <>
        Arranged in a tree-indexed manner with search functionality, enable you to easily discover the relevant content.
      </>
    ),
  },
  {
    title: 'Keeps you updated',
    Svg: require('../../static/img/front-bird.svg').default,
    description: (
      <>
        Recohut documentation is latest and covers all kinds of new developments in recsys community.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
