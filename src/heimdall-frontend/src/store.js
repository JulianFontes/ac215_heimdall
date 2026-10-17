import {legacy_createStore as createStore, combineReducers, applyMiddleware, compose} from 'redux';
import thunk from 'redux-thunk';
import keplerGlReducer, {enhanceReduxMiddleware} from '@kepler.gl/reducers';

const reducers = combineReducers({
  keplerGl: keplerGlReducer.initialState({
    // open the filter panel by default so the confidence histogram is visible
    uiState: {activeSidePanel: 'filter'}
  })
});

// enhanceReduxMiddleware appends react-palm's taskMiddleware itself, so don't add it again
const middlewares = enhanceReduxMiddleware([thunk]);

export default createStore(reducers, {}, compose(applyMiddleware(...middlewares)));