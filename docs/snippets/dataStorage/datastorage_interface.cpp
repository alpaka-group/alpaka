#include <alpaka/alpaka.hpp>

// the global objects and by-value parameters are what this documentation example shows
// NOLINTBEGIN(bugprone-throwing-static-initialization, performance-unnecessary-value-param)
int ptr = 0;

alpaka::LinearizedIdxGenerator otherLinearizedIdxGenerator{alpaka::Vec{0}};
alpaka::MdSpan otherMdspan{&ptr, alpaka::Vec{size_t{1}}, alpaka::Vec{sizeof(int)}};
alpaka::View otherView{alpaka::api::host, &ptr, alpaka::Vec{size_t{1}}, alpaka::Vec{sizeof(int)}};
alpaka::onHost::SharedBuffer otherSharedBuffer = alpaka::onHost::allocHost<int>(alpaka::Vec{1});

// BEGIN-DATASTORAGE-interface
void func1(alpaka::concepts::IDataSource auto input)
{
    alpaka::unused(input);
}

void func2(alpaka::concepts::IView auto input)
{
    alpaka::unused(input);
}

int main()
{
    // fulfill the IDataSource interface, but not the IMdSpan interface
    alpaka::LinearizedIdxGenerator linearizedIdxGenerator = otherLinearizedIdxGenerator;
    // fulfil the IDataSource and the IMdSpan interface, but not the IView interface
    alpaka::MdSpan mdpsan = otherMdspan;
    // fulfil the IDataSource, the IMdSpan and the IView interface, but not the IBuffer interface
    alpaka::View view = otherView;
    // fulfil the IDataSource, the IMdSpan, the IView and the IBuffer interface
    alpaka::onHost::SharedBuffer sharedBuffer = otherSharedBuffer;

    func1(linearizedIdxGenerator);
    func1(mdpsan);
    func1(view);
    func1(sharedBuffer);

    // does not compile, linearizedIdxGenerator does not fulfil the IView interface
    // func2(linearizedIdxGenerator);
    // does not compile, mdspan does not fulfil the IView interface
    // func2(mdpsan);
    func2(view);
    func2(sharedBuffer);
    return 0;
}

// END-DATASTORAGE-interface
// NOLINTEND(bugprone-throwing-static-initialization, performance-unnecessary-value-param)
