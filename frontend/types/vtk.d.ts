/**
 * vtk.js rendering profiles are side-effect-only modules: importing one registers
 * the OpenGL view-node factories for a class of renderables. They ship no type
 * declarations because they export nothing, so TypeScript needs to be told they
 * exist.
 *
 * Without the Volume profile imported, vtk.js raises "No vtkOpenGLViewNodeFactory
 * implementation found for vtkRenderer" and no volume is drawn.
 */
declare module "@kitware/vtk.js/Rendering/Profiles/Volume";
declare module "@kitware/vtk.js/Rendering/Profiles/Geometry";
declare module "@kitware/vtk.js/Rendering/Profiles/All";
